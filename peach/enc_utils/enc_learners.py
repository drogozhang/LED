import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from peach.enc_utils.sim_metric import Similarity
from peach.enc_utils.general import preproc_inputs

from transformers.modeling_utils import PreTrainedModel

from peach.nn_utils.optimal_trans import ipot, trace, optimal_transport_dist
from peach.nn_utils.general import mask_out_cls_sep


class FLOPS(nn.Module):
    """constraint from Minimizing FLOPs to Learn Efficient Sparse Representations
    https://arxiv.org/abs/2004.05665
    """
    def __init__(self):
        super().__init__()

    def forward(self, batch_rep):
        return torch.sum(torch.mean(torch.abs(batch_rep), dim=0) ** 2)


class LearnerMixin(PreTrainedModel):
    def __init__(self, model_args, config, tokenizer, encoder, query_encoder=None, ):
        super().__init__(config)

        self.model_args = model_args
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.query_encoder = query_encoder

    @property
    def encoding_doc(self):
        return self.encoder

    @property
    def encoding_query(self):
        if self.query_encoder is None:
            return self.encoder
        return self.query_encoder

    @property
    def my_world_size(self):
        if dist.is_initialized():
            world_size = dist.get_world_size()
        else:
            world_size = 1
        return world_size

    def gather_tensor(self, target_tensor):
        if dist.is_initialized() and dist.get_world_size() > 1 and self.training:
            target_tensor_list = [torch.zeros_like(target_tensor) for _ in range(dist.get_world_size())]
            dist.all_gather(tensor_list=target_tensor_list, tensor=target_tensor.contiguous())
            target_tensor_list[dist.get_rank()] = target_tensor
            target_tensor_gathered = torch.cat(target_tensor_list, 0)
        else:
            target_tensor_gathered = target_tensor
        return target_tensor_gathered

    @staticmethod
    def _world_size():
        if dist.is_initialized():
            return dist.get_world_size()
        else:
            return 1


def add_training_hyperparameters(parser):
    parser.add_argument("--lambda_d", type=float, default=0.0008)
    parser.add_argument("--lambda_q", type=float, default=0.0006)
    parser.add_argument("--lambda_gamma", type=float, default=0.5)
    parser.add_argument("--apply_distill", action="store_true")

    parser.add_argument(
        "--consistency_method", type=str, default=None, choices=["fix", "len", "dist"])
    parser.add_argument("--consistency_delta", type=float, default=1.0)
    parser.add_argument("--consistency_loss_weight", type=float, default=1.0)

    parser.add_argument("--hidden_method", type=str, default=None, )
    parser.add_argument("--hidden_margin", type=float, default=1.0)
    parser.add_argument("--hidden_loss_weight", type=float, default=1.0)

    parser.add_argument("--ipot_method", type=str, default=None, choices=["naive", "weighted"])
    parser.add_argument("--ipot_loss_type", type=str, default="distance", choices=["distance", "margin", "xentropy"])
    parser.add_argument("--ipot_loss_weight", type=float, default=1.0)
    parser.add_argument("--do_ipot_norm", action="store_true")

    parser.add_argument("--weight_method", type=str, default=None, choices=["wei1", "wei2", "wei3"])

    # tricks

    parser.add_argument("--apply_mlm", action="store_true")
    parser.add_argument("--mlm_prob", type=float, default=0.25)
    parser.add_argument("--local_mlm_loss_weight", type=float, default=0.1)
    # parser.add_argument("--global_mlm_loss_weight", type=float, default=0.0)

    return []  # this only for enc's config file


from peach.nn_utils.general import mask_token_random
from peach.nn_utils.general import exp_mask, zero_mask

class SpladeEncoderLearner(LearnerMixin):
    def __init__(self, model_args, config, tokenizer, encoder, query_encoder=None):
        super().__init__(model_args, config, tokenizer, encoder, query_encoder, )

        self.sim_fct = Similarity(metric="dot")
        self.flops_loss = FLOPS()

    def forward(
            self,
            input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None,
            input_ids_query=None, attention_mask_query=None, token_type_ids_query=None, position_ids_query=None,
            distill_labels=None, training_progress=None,
            training_mode=None,
            **kwargs,
    ):

        if training_mode is None:
            return self.encoder(input_ids, attention_mask, ).contiguous()

        dict_for_meta = {}

        # input reshape
        (high_dim_flag, num_docs), (org_doc_shape, org_query_shape), \
        (input_ids, attention_mask, token_type_ids, position_ids,), \
        (input_ids_query, attention_mask_query, token_type_ids_query, position_ids_query,) = preproc_inputs(
            input_ids, attention_mask, token_type_ids, position_ids,
            input_ids_query, attention_mask_query, token_type_ids_query, position_ids_query,)
        bsz = org_doc_shape[0]

        if self.model_args.apply_mlm and self.training:
            input_ids, mlm_labels = mask_token_random(
                input_ids, attention_mask, prob=self.model_args.mlm_prob,
                mask_token_id=self.tokenizer.mask_token_id)

        # encoding
        doc_outputs = self.encoding_doc(input_ids, attention_mask, return_dict=True)
        query_outputs = self.encoding_query(input_ids_query, attention_mask_query, return_dict=True)

        emb_doc, emb_query = doc_outputs["sparse_embedding"], query_outputs["sparse_embedding"]
        emb_dim = emb_doc.shape[-1]

        dict_for_loss = {}
        # calculate similarity and losses
        ga_emb_doc, ga_emb_query = self.gather_tensor(emb_doc), self.gather_tensor(emb_query)

        # if dist.is_initialized():
        dict_for_meta["flops_doc_loss_weight"] = self.model_args.lambda_d
        dict_for_meta["flops_query_loss_weight"] = self.model_args.lambda_q
        dict_for_meta["sparse_xentropy_loss_weight"] = float(self.my_world_size)

        loss_fct = nn.CrossEntropyLoss()
        target = torch.arange(ga_emb_query.shape[0], device=ga_emb_query.device, dtype=torch.long) * num_docs

        similarities = self.sim_fct(ga_emb_query, ga_emb_doc.view(-1, emb_dim))
        dict_for_loss["sparse_xentropy_loss"] = loss_fct(similarities, target)
        # sparse
        dict_for_loss["flops_doc_loss"] = self.flops_loss(emb_doc.view(-1, emb_dim))
        dict_for_loss["flops_query_loss"] = self.flops_loss(emb_query)

        if self.model_args.apply_mlm:
            dict_for_meta["local_mlm_loss_weight"] = self.model_args.local_mlm_loss_weight
            dict_for_loss["local_mlm_loss"] = nn.CrossEntropyLoss(ignore_index=-100)(
                doc_outputs["prediction_logits"].view(-1, doc_outputs["prediction_logits"].shape[-1]), mlm_labels.view(-1))

            # token_level prediction -> global sparse vector todo: add to new proj
            # if self.model_args.global_mlm_loss_weight > 1e-4:
            #     raise NotImplementedError
            #     dict_for_meta["global_mlm_loss_weight"] = self.model_args.global_mlm_loss_weight
            #     dense_similarities = (doc_outputs["hidden_states"] * doc_outputs["dense_embedding"].unsqueeze(-2)).sum(-1)  # [bs*nd,dsl,d] -> [bs*nd,dsl]
            #     doc_seq_len = dense_similarities.shape[-2]
            #
            #     top_values, target_ids = torch.topk(doc_outputs["sparse_embedding"], k=200)  # [bs*nd, 200]  # todo: add self
            #     top_values = top_values.unsqueeze(-2).repeat(1, doc_seq_len, 1)  # [bs*nd, dsl, 200]
            #     target_ids = target_ids.unsqueeze(-2).repeat(1, doc_seq_len, 1)  # [bs*nd, dsl, 200]
            #
            #     top_mask = top_values > torch.log(1. + torch.relu(dense_similarities))  # [bs*nd, dsl, 200]
            #     top_values = zero_mask(top_mask, top_values, high_rank=False)  # [bs*nd,dsl,200]
            #     norm_top_values = F.normalize(top_values, p=1, dim=-1)# [bs*nd,dsl,200]
            #
            #     log_probs = torch.log_softmax(doc_outputs["prediction_logits"], dim=-1)  # [bs*nd, dsl, V]
            #     ga_log_probs = torch.gather(log_probs, dim=-1, index=target_ids)  # [bs*nd, dsl, 200]
            #     global_mlm_losses = (norm_top_values * ga_log_probs).sum(-1)  # [bs*nd,dsl]
            #     dict_for_loss["global_mlm_loss"] = global_mlm_losses[mlm_labels >= 0].mean()

        # semantic consistency
        den_dim = doc_outputs["hidden_states"].shape[-1]
        if self.model_args.consistency_method is not None:
            dict_for_meta["consistent_doc_loss_weight"] = self.model_args.consistency_loss_weight
            dict_for_meta["consistent_query_loss_weight"] = self.model_args.consistency_loss_weight
            consistency_delta = self.model_args.consistency_delta
            dict_for_loss["consistent_doc_loss"] = self.get_semantic_consistency(
                doc_outputs["hidden_states"], doc_outputs["attention_mask"], delta=consistency_delta)
            dict_for_loss["consistent_query_loss"] = self.get_semantic_consistency(
                query_outputs["hidden_states"], query_outputs["attention_mask"], delta=consistency_delta)

        if self.model_args.hidden_method is not None:
            dict_for_meta["hidden_loss_weight"] = self.model_args.hidden_loss_weight
            dict_for_loss["hidden_loss"] = self.get_hidden_formulate(
                query_outputs["hidden_states"], query_outputs["attention_mask"],
                doc_outputs["hidden_states"].view(bsz, num_docs, -1, den_dim),
                doc_outputs["attention_mask"].view(bsz, num_docs, -1),
                query_outputs, doc_outputs
            )

        if self.model_args.ipot_method is not None:
            dict_for_meta["ipot_loss_weight"] = self.model_args.ipot_loss_weight

            ipot_loss, ipot_meta = self.get_ipot(
                query_outputs["hidden_states"], query_outputs["attention_mask"],
                doc_outputs["hidden_states"].view(bsz, num_docs, -1, den_dim),
                doc_outputs["attention_mask"].view(bsz, num_docs, -1),
                use_cos=True, max_num_negs=2, eps=1e-4,
                query_outputs=query_outputs, doc_outputs=doc_outputs,
            )
            dict_for_loss["ipot_loss"] = ipot_loss
            dict_for_meta.update(ipot_meta)

        # prepare sparse reture
        # # seq-level and token-level
        qry_seq_semb, qry_tk_semb = query_outputs["sparse_embedding"], query_outputs[
            "saturated_token_embeddings"]
        doc_seq_semb, doc_tk_semb = doc_outputs["sparse_embedding"], doc_outputs["saturated_token_embeddings"]
        doc_seq_semb = doc_seq_semb.view(bsz, num_docs, emb_dim)
        doc_tk_semb = doc_tk_semb.view(bsz, num_docs, -1, emb_dim)

        dict_for_meta.update({
            "qry_seq_semb": qry_seq_semb, "qry_tk_semb": qry_tk_semb,
            "doc_seq_semb": doc_seq_semb, "doc_tk_semb": doc_tk_semb,})

        """
        parser.add_argument("--detach_kl_target", action="store_true")
        parser.add_argument("--dense_xentropy_loss_weight", type=float, default=1.0)
        parser.add_argument("--agree_kl_loss_weight", type=float, default=1.0)
        # den_splade part
        denemb_doc, denemb_query = doc_outputs["dense_embedding"], query_outputs["dense_embedding"]
        denemb_dim = denemb_doc.shape[-1]
        ga_denemb_doc, ga_denemb_query = self.gather_tensor(denemb_doc), self.gather_tensor(denemb_query)
        dict_for_meta["dense_xentropy_loss_weight"] = float(
            self.my_world_size) * self.model_args.dense_xentropy_loss_weight
        dict_for_meta["agree_kl_loss_weight"] = self.model_args.agree_kl_loss_weight

        den_similarities = self.sim_fct(ga_denemb_query, ga_denemb_doc.view(-1, denemb_dim))
        dict_for_loss["dense_xentropy_loss"] = loss_fct(den_similarities, target)
        # hard negs only for KL
        neg_sims = self.sim_fct.forward_qd_pair(emb_query.unsqueeze(1), emb_doc.view(bsz, num_docs, -1))
        den_neg_sims = self.sim_fct.forward_qd_pair(denemb_query.unsqueeze(1), denemb_doc.view(bsz, num_docs, -1))

        kl_fct = nn.KLDivLoss(reduction="batchmean")
        kl_target = torch.softmax(den_neg_sims, dim=-1)
        if self.model_args.detach_kl_target:
            kl_target = kl_target.detach()
        dict_for_loss["agree_kl_loss"] = kl_fct(torch.log_softmax(neg_sims, dim=-1), kl_target)
        """

        # # sparse losses
        #
        # if hasattr(self.encoder, "embedding_dim") and self.encoder.embedding_dim > self.encoder.config.vocab_size:
        #     dict_for_loss["flops_doc_loss"] = self.flops_loss(emb_doc.view(-1, emb_dim)[:, :self.encoder.config.vocab_size])
        #     dict_for_loss["flops_query_loss"] = self.flops_loss(emb_query[:, :self.encoder.config.vocab_size])
        #
        #     dict_for_meta["exflops_doc_loss_weight"] = self.model_args.lambda_d * self.model_args.lambda_gamma
        #     dict_for_meta["exflops_query_loss_weight"] = self.model_args.lambda_q * self.model_args.lambda_gamma
        #     dict_for_loss["exflops_doc_loss"] = self.flops_loss(emb_doc.view(-1, emb_dim)[:, self.encoder.config.vocab_size:])
        #     dict_for_loss["exflops_query_loss"] = self.flops_loss(emb_query[:, self.encoder.config.vocab_size:])

        # distillation
        if self.model_args.apply_distill and distill_labels is not None:
            bi_scores = self.sim_fct.forward_qd_pair(
                emb_query.unsqueeze(1), emb_doc.view(bsz, num_docs, -1), )  # bs,nd
            dict_for_loss["distill_ce_loss"] = nn.KLDivLoss()(
                torch.log_softmax(bi_scores, dim=-1), torch.softmax(distill_labels, dim=-1))

        loss = 0.
        for k in dict_for_loss:
            if k + "_weight" in dict_for_meta:
                loss += dict_for_meta[k + "_weight"] * dict_for_loss[k]
            else:
                loss += dict_for_loss[k]
        dict_for_loss["loss"] = loss

        dict_for_meta.update(dict_for_loss)
        return dict_for_meta

    @staticmethod
    def get_sparse_salience(seq_semb, tk_semb, attn_mask=None, weight_method=None, ):
        # seq_semb [..., dim], tk_semb [..., sl, dim], attn_mask [..., sl]

        dtype = seq_semb.dtype

        if weight_method == "wei1":  #
            token_weights = (tk_semb / (seq_semb + 1e-4).unsqueeze(-2)).sum(-1) / \
                                ((seq_semb > 0.).to(dtype).sum(-1).unsqueeze(-1) + 1e-4)  # bs,sl
        elif weight_method == "wei2":
            token_weights = tk_semb.sum(-1) / (seq_semb.sum(-1).unsqueeze(-1) + 1e-4)
        else:
            raise NotImplementedError

        if attn_mask is not None:
            token_weights = token_weights * attn_mask.to(dtype)

        return token_weights

    def get_ipot(
            self, qry_hiddens, qry_attn_mask, doc_hiddens, doc_attn_mask,
            use_cos=True, max_num_negs=1, eps=1e-4,
            query_outputs=None, doc_outputs=None,
            **kwargs):
        ipot_meta_dict = dict()

        bs, org_nd, dsl, hn = doc_hiddens.shape
        qsl = qry_hiddens.shape[1]
        device, dtype = doc_hiddens.device, doc_hiddens.dtype

        if org_nd > max_num_negs + 1:
            nd = max_num_negs + 1
            doc_hiddens = doc_hiddens[:, :nd].contiguous()
            doc_attn_mask = doc_attn_mask[:, :nd].contiguous()
        else:
            nd = org_nd

        if use_cos:  # [bs,qsl,hn] vs. [bs,nd,dsl,hn]
            qry_hiddens = F.normalize(qry_hiddens, p=2, dim=-1, eps=eps)
            doc_hiddens = F.normalize(doc_hiddens, p=2, dim=-1, eps=eps)

        # pad qry for multi-docs
        qry_hiddens = qry_hiddens.unsqueeze(1).repeat(1, nd, 1, 1).contiguous()
        qry_attn_mask = qry_attn_mask.unsqueeze(1).repeat(1, nd, 1).contiguous()

        # flat bs,nd for calculation
        qry_hiddens = qry_hiddens.view(bs*nd, qsl, hn)
        qry_attn_mask = qry_attn_mask.view(bs*nd, qsl)
        doc_hiddens = doc_hiddens.view(bs*nd, dsl, hn)
        doc_attn_mask = doc_attn_mask.view(bs*nd, dsl)

        # begin to calculate
        qry_mask, doc_mask = mask_out_cls_sep(qry_attn_mask), mask_out_cls_sep(doc_attn_mask)  # remove cls and sep
        qry_mask_bl, doc_mask_bl = qry_mask.bool(), doc_mask.bool()
        qry_pad_bl, doc_pad_bl = ~qry_mask_bl, ~doc_mask_bl
        joint_pad = qry_pad_bl.unsqueeze(-1) | doc_pad_bl.unsqueeze(-2)  # pad tokens

        with torch.cuda.amp.autocast(enabled=False):
            if self.model_args.ipot_method == "weighted":
                qry_seq_semb, qry_tk_semb = query_outputs["sparse_embedding"], query_outputs[
                    "saturated_token_embeddings"]
                doc_seq_semb, doc_tk_semb = doc_outputs["sparse_embedding"], doc_outputs["saturated_token_embeddings"]

                doc_seq_semb = doc_seq_semb.view(bs, org_nd, -1)[:, :nd]
                doc_tk_semb = doc_tk_semb.view(bs, org_nd, dsl, -1)[:, :nd]

                qry_token_weights = self.get_sparse_salience(  # bs,qsl. [NOTE: qry_mask [bs,nd,qsl]]
                    qry_seq_semb, qry_tk_semb, qry_mask.view(bs,nd,qsl)[:,0], weight_method=self.model_args.weight_method)
                qry_token_weights = qry_token_weights.unsqueeze(1).repeat(1, nd, 1).contiguous()  # [bs,nd,qsl]

                doc_token_weights = self.get_sparse_salience(  # bs,nd,dsl
                    doc_seq_semb, doc_tk_semb, doc_mask, weight_method=self.model_args.weight_method)

                qry_token_weights = F.normalize(qry_token_weights, p=1, dim=-1, eps=eps)
                doc_token_weights = F.normalize(doc_token_weights, p=1, dim=-1, eps=eps)
            else:
                qry_token_weights, doc_token_weights = None, None

            cost = 1 - torch.matmul(qry_hiddens, doc_hiddens.transpose(-1, -2))  # [bs*nd, qsl, dsl]
            cost.masked_fill_(joint_pad, 0)  # assign sim mat with 0 for

            qry_len = qry_mask_bl.to(dtype).sum(-1)  # [bs*nd]
            doc_len = doc_mask_bl.to(dtype).sum(-1)  # [bs*nd]

            ot_mat = ipot(  # [bs*nd,dsl,qsl]
                cost.detach(), qry_len, qry_pad_bl, doc_len, doc_pad_bl, joint_pad, 0.5, 50, 1)
            if doc_token_weights is not None:
                ot_mat = ot_mat * doc_token_weights.view(bs*nd, dsl, 1)  # []
            if self.model_args.do_ipot_norm:
                ot_mat = F.normalize(ot_mat, p=1, dim=-2, eps=eps)

            # naive distance-based loss
            # pre_distance = cost.matmul(ot_mat.detach())  # [bs*nd, qsl, qsl]
            ot_distances = (cost * ot_mat.transpose(-2, -1)).sum(-1)  # [bs*nd, qsl]
            if doc_token_weights is not None:
                ipot_meta_dict["qry_token_weights"], ipot_meta_dict["doc_token_weights"] = qry_token_weights[:, 0], doc_token_weights
            ipot_meta_dict["ot_cost"] = cost.view(bs, nd, qsl, dsl)  # [bs,nd,dsl,qsl]
            ipot_meta_dict["ot_mat"] = ot_mat.view(bs, nd, dsl, qsl)  # [bs,nd,dsl,qsl]
            # ipot_meta_dict["ot_pre_distance"] = pre_distance.view(bs, nd, qsl, qsl)  #
            ipot_meta_dict["ot_distances"] = ot_distances  #  torch.diagonal(pre_distance, dim1=-2, dim2=-1).view(bs, nd, qsl)

        if self.model_args.ipot_loss_type in ["distance", "margin", "xentropy"]:
            if qry_token_weights is not None:
                ot_distances = ot_distances * qry_token_weights.view(bs*nd, qsl)
            distance = ot_distances.sum(-1).view(bs, nd)
            if self.model_args.ipot_loss_type == "distance":
                ot_loss = distance[:, 0].mean() - distance[:, 1:].mean() # distance-based
            elif self.model_args.ipot_loss_type == "margin":
                ot_loss = torch.relu(0.5 - distance[:, :1] + distance[:, 1:]).mean()  # margin-based
            elif self.model_args.ipot_loss_type == "xentropy":
                ot_loss = nn.CrossEntropyLoss(ignore_index=-100)(  # xentropy-based
                    distance, torch.zeros([bs], dtype=torch.long, device=device))
            else:
                raise NotImplementedError
        elif self.model_args.ipot_loss_type in ["tk_xentropy"]:
            ot_distances_rsp = ot_distances.view(bs, nd, qsl)  # [bs, qsl, nd]
            if self.model_args.ipot_loss_type == "tk_xentropy":
                labels = -100 * (1 - qry_mask.view(bs, nd, qsl)[:, 0])  # bs,qsl \in {0, -100}
                xentropy_fct = nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
                losses = xentropy_fct(ot_distances_rsp.transpose(-1, -2).contiguous().view(-1, nd), labels.view(-1)).view(bs, qsl)
            else:
                raise NotImplementedError
            ot_loss = (losses.view(bs, qsl) * qry_token_weights[:, 0]).sum(-1).mean()
        else:
            raise NotImplementedError

        return ot_loss, ipot_meta_dict

    def get_semantic_consistency(self, hiddens, attn_mask, delta,):
        bs, sl, hn = hiddens.shape
        device, dtype = hiddens.device, hiddens.dtype

        hd_sim = torch.matmul(hiddens, hiddens.transpose(-1, -2))  # bs,sl,sl
        max_val = torch.diagonal(hd_sim, dim1=1, dim2=2).unsqueeze(-1)  # bs,sl,1
        attn_mask_ft = attn_mask.to(dtype)

        if self.model_args.consistency_method == "fix":
            deltas = delta
        elif self.model_args.consistency_method == "len":
            deltas = (attn_mask_ft.sum(-1) ** 0.5) * delta
        elif self.model_args.consistency_method == "dist":

            al = torch.arange(sl, device=device)
            distances = torch.abs(al.view(1,sl) - al.view(sl,1).to(dtype))  # sl,sl
            deltas = (torch.log(1.+distances) * delta).view(1,sl,sl).repeat(bs,1,1)
        else:
            raise NotImplementedError

        mask_3d_ft = attn_mask_ft.unsqueeze(-1) * attn_mask_ft.unsqueeze(-2)  # bs,sl,sl

        losses = torch.relu(max_val - deltas - hd_sim) * mask_3d_ft
        loss = losses.sum() / mask_3d_ft.sum()
        return loss

    def get_hidden_formulate(self, qry_hiddens, qry_attn_mask, doc_hiddens, doc_attn_mask, query_outputs, doc_outputs, **kwargs):
        """qry_hiddens[bs,qsl,hn], qry_attn_mask[bs,qsl], doc_hiddens[bs,nd,dsl,hn], doc_attn_mask[bs,nd,dsl]"""
        bs, nd, dsl, hn = doc_hiddens.shape
        qsl = qry_hiddens.shape[1]
        device, dtype = doc_hiddens.device, doc_hiddens.dtype

        # gold_doc_hiddens, gold_doc_attn_mask = doc_hiddens[:, 0].contiguous(), doc_attn_mask[:, 0].contiguous()
        # gold_qd_sims = torch.matmul(qry_hiddens, gold_doc_hiddens)  # bs,qsl,dsl

        if self.model_args.hidden_method in [
            "wei1_qmax", "wei2_qmax",
            "wei1dtc_qmax", "wei2dtc_qmax",
            "wei1_qmaxpool", "wei2_qmaxpool"]:
            # bs,sl
            if self.model_args.hidden_method.startswith("wei1"):
                qry_token_weights = (query_outputs["saturated_token_embeddings"] / (query_outputs["sparse_embedding"]+1e-4).unsqueeze(1)).sum(-1) / \
                    ((query_outputs["sparse_embedding"] > 0.).sum(-1).to(dtype).unsqueeze(-1) + 1e-4)  # bs,qsl
                doc_token_weights = (doc_outputs["saturated_token_embeddings"] / (doc_outputs["sparse_embedding"]+1e-4).unsqueeze(1)).sum(-1) / \
                    ((doc_outputs["sparse_embedding"] > 0.).sum(-1).to(dtype).unsqueeze(-1) + 1e-4)  # bs,qsl
            else:
                qry_token_weights = query_outputs["saturated_token_embeddings"].sum(-1) / (
                        query_outputs["sparse_embedding"].sum(-1).unsqueeze(1) + 1e-4)
                doc_token_weights = doc_outputs["saturated_token_embeddings"].sum(-1) / (  # bs*nd,dl
                        doc_outputs["sparse_embedding"].sum(-1).unsqueeze(1) + 1e-4)
            doc_token_weights = doc_token_weights.view(bs, nd, dsl)

            if "dtc" in self.model_args.hidden_method:
                qry_token_weights = qry_token_weights.detach()
                doc_token_weights = doc_token_weights.detach()

            # print("qry_token_weights", torch.isnan(qry_token_weights).sum(), torch.isinf(qry_token_weights).sum())
            # print("doc_token_weights", torch.isnan(doc_token_weights).sum(), torch.isinf(doc_token_weights).sum())

            masked_qry_token_weights = qry_token_weights * qry_attn_mask.to(dtype)  # bs,qsl
            norm_qry_token_weights = masked_qry_token_weights / (torch.sum(masked_qry_token_weights, dim=-1, keepdim=True) + 1e-4)
            # print("norm_qry_token_weights", torch.isnan(norm_qry_token_weights).sum(), torch.isinf(norm_qry_token_weights).sum())

            qd_sims = torch.matmul(qry_hiddens.unsqueeze(1), doc_hiddens.transpose(-1, -2))  # bs,nd,qsl,dsl
            qd_sims = qd_sims - 10000. * (1 - doc_attn_mask).to(dtype).unsqueeze(-2)  # masking for masking&maxpool

            pooled_qd_sims, pooled_qd_idxs = qd_sims.max(-1)  # [bs,nd,qsl]
            max_idx_weights = torch.gather(doc_token_weights, dim=2, index=pooled_qd_idxs)  # [bs,nd,qsl]

            logits = torch.log(max_idx_weights + 1e-7) + pooled_qd_sims  # bs,nd,qsl
            # print("logits", torch.isnan(logits).sum(), torch.isinf(logits).sum())

            if "pool" not in self.model_args.hidden_method:
                labels = -100 * (1 - qry_attn_mask)  # [bs,qsl] 0 or -100
                xentropy_fct = nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
                losses = xentropy_fct(logits.transpose(-1, -2).contiguous().view(-1, nd), labels.view(-1))
                loss = (losses.view(bs, qsl) * norm_qry_token_weights).sum(-1).mean()
            else:
                logits = (logits * norm_qry_token_weights.unsqueeze(1)).sum(-1)  # bs,nd,qsl -> bs,nd
                xentropy_fct = nn.CrossEntropyLoss(ignore_index=-100)
                loss = xentropy_fct(logits, torch.zeros([bs], dtype=torch.long, device=device))

        elif self.model_args.hidden_method == "qmax":
        # query-ware hiddens
            qd_sims = torch.matmul(qry_hiddens.unsqueeze(1), doc_hiddens.transpose(-1, -2))  # bs,nd,qsl,dsl
            qd_sims = qd_sims - 10000. * (1 - doc_attn_mask).to(dtype).unsqueeze(-2)  # masking for masking&maxpool

            pooled_qd_sims = qd_sims.max(-1)[0]  # bs,nd,qsl

            # gold_qd_sims = pooled_qd_sims[:, :1]  # bs,1,qsl
            # neg_qd_sims = pooled_qd_sims[:, 1:]  # bs,nd-1,qsl
            # losses = torch.relu(self.model_args.hidden_margin - gold_qd_sims + neg_qd_sims)# bs,nd-1,qsl
            # extend_qry_mask_ft = qry_attn_mask.unsqueeze(1).to(dtype).repeat(1, nd-1, 1)# bs,nd-1,qsl
            # loss = (losses * extend_qry_mask_ft).sum() / extend_qry_mask_ft.sum()

            labels = -100 * (1 - qry_attn_mask)  # [bs,qsl] 0 or -100
            xentropy_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = xentropy_fct(pooled_qd_sims.transpose(-1, -2).contiguous().view(-1, nd), labels.view(-1))
        elif self.model_args.hidden_method == "qmaxpool":
            qd_sims = torch.matmul(qry_hiddens.unsqueeze(1), doc_hiddens.transpose(-1, -2))  # bs,nd,qsl,dsl
            qd_sims = qd_sims - 10000. * (1 - doc_attn_mask).to(dtype).unsqueeze(-2)  # masking for masking&maxpool

            pooled_qd_sims = qd_sims.max(-1)[0]  # bs,nd,qsl

            qry_pool_mask = qry_attn_mask.unsqueeze(1).to(dtype).repeat(1,nd,1)  # bs,nd,qsl
            pooled_scores = (pooled_qd_sims * qry_pool_mask).sum(-1) / qry_pool_mask.sum(-1)  # bs,nd
            xentropy_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = xentropy_fct(pooled_scores, torch.zeros([bs], dtype=torch.long, device=device))
        elif self.model_args.hidden_method == "qatt":
            qd_sims = torch.matmul(qry_hiddens.unsqueeze(1), doc_hiddens.transpose(-1, -2))  # bs,nd,qsl,dsl
            qd_sims = qd_sims - 10000. * (1 - doc_attn_mask).to(dtype).unsqueeze(-2)  # masking for masking&maxpool
            qd_probs = torch.softmax(qd_sims, dim=-1)  # [bs,nd,qsl,dsl]
            att_qry_hiddens = torch.matmul(qd_probs, doc_hiddens)  # [bs,nd,qsl,hn]

            # torch.matmul(qry_hiddens.unsqueeze(1), att_qry_hiddens.transpose(-1, -2))  # bs,nd,qsl,qsl
            logits = (att_qry_hiddens * qry_hiddens.unsqueeze(1)).sum(-1)  # [bs,nd,qsl]
            logits = logits.transpose(-1, -2).contiguous()  # [bs,qsl,nd]

            labels = -100 * (1 - qry_attn_mask)  # [bs,qsl] 0 or -100
            xentropy_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = xentropy_fct(logits.view(-1, nd), labels.view(-1))
        elif self.model_args.hidden_method == "qattpool":
            qd_sims = torch.matmul(qry_hiddens.unsqueeze(1), doc_hiddens.transpose(-1, -2))  # bs,nd,qsl,dsl
            qd_sims = qd_sims - 10000. * (1 - doc_attn_mask).to(dtype).unsqueeze(-2)  # masking for masking&maxpool
            qd_probs = torch.softmax(qd_sims, dim=-1)  # [bs,nd,qsl,dsl]
            att_qry_hiddens = torch.matmul(qd_probs, doc_hiddens)  # [bs,nd,qsl,hn]

            # torch.matmul(qry_hiddens.unsqueeze(1), att_qry_hiddens.transpose(-1, -2))  # bs,nd,qsl,qsl
            logits = (att_qry_hiddens * qry_hiddens.unsqueeze(1)).sum(-1).contiguous()  # [bs,nd,qsl]

            qry_pool_mask = qry_attn_mask.unsqueeze(1).to(dtype).repeat(1, nd, 1)  # bs,nd,qsl
            pooled_scores = (logits * qry_pool_mask).sum(-1) / qry_pool_mask.sum(-1)  # bs,nd
            xentropy_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = xentropy_fct(pooled_scores, torch.zeros([bs], dtype=torch.long, device=device))
        else:
            raise NotImplementedError


        return loss

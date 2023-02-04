# distill on the fly

import collections
import json
from random import choices
from secrets import choice
import tqdm
from itertools import combinations

from peach.base import *

from peach.datasets.marco.dataset_marco_passages import DatasetMarcoPassagesRanking
from peach.datasets.marco.dataset_marco_eval import DatasetRerank, DatasetCustomRerank

from peach.enc_utils.eval_functions import evaluate_encoder_reranking
from peach.enc_utils.eval_dense import evaluate_dense_retreival
from peach.enc_utils.hn_gen_dense import get_hard_negative_by_dense_retrieval
from peach.enc_utils.general import get_representation_tensor
from peach.nn_utils.general import combine_dual_inputs_by_attention_mask

from proj_dense.train_dense_retriever import train
from proj_sparse.modeling_splade_series import add_model_hyperparameters, DistilBertSpladeEnocder, \
    BertSpladeEnocder, ConSpladeEnocder, RobertaSpladeEnocder

from transformers import AutoModel, AutoModelForSequenceClassification


from peach.enc_utils.enc_learners import LearnerMixin
from proj_dense.train_dense_retriever import DenseLearner

import torch
import torch.nn as nn

from peach.common import load_pickle, file_exists
from peach.enc_utils.sim_metric import Similarity
from peach.enc_utils.general import preproc_inputs


def add_training_hyperparameters(parser):
    parser.add_argument("--apply_dst", action="store_true")
    parser.add_argument("--apply_inbatch", action="store_true")
    parser.add_argument("--with_ce_loss", action="store_true")
    
    parser.add_argument("--use_addition_qrels", action="store_true")

    parser.add_argument("--dst_loss_weight", type=float, default=1.0)

    parser.add_argument("--sp_tch_model_path", type=str, default=None)
    parser.add_argument("--sp_tch_encoder", type=str, default='distilbert')
    parser.add_argument("--dst_method", type=str, default='kl', choices=['kl', 'rank', 'ranknet', 'listnet'])

    parser.add_argument("--tch_no_drop", action="store_true")

    parser.add_argument("--xe_tch_model_path", type=str, default=None)
    parser.add_argument("--xe_tch_encoder", type=str, default='xencoder')
    parser.add_argument("--xe_tch_dst_loss_weight", type=float, default=1.0)
    parser.add_argument("--xe_tch_temp", type=float, default=1.0)

    parser.add_argument("--with_pos", action='store_true')

    pass


class DenseLearnerWithSpladeDistil(DenseLearner):
    def __init__(self, model_args, config, tokenizer, encoder, sparse_encoder, xencoder, sparse_query_encoder=None, query_encoder=None):
        super().__init__(model_args, config, tokenizer, encoder, query_encoder, )
        self.sparse_encoder = sparse_encoder
        self.sparse_query_encoder = sparse_query_encoder
        self.xe_tch_encoder = xencoder
    
    @property
    def sparse_encoding_doc(self):
        return self.sparse_encoder

    @property
    def sparse_encoding_query(self):
        if self.sparse_query_encoder is None:
            return self.sparse_encoder
        return self.sparse_query_encoder
    
    @staticmethod
    def mask_out_logits(sims, mask):
        mask = mask.to(sims.dtype)
        return sims * (1 - mask) - 10000 * mask

    def get_delta(self, bsz, num_docs):
        return dist.get_rank() * bsz * num_docs if self.my_world_size > 1 else 0

    def generate_positive_mask(self, bsz, num_docs, do_in_batch, device, ):
        dtype = torch.long
        with torch.no_grad():
            if do_in_batch:
                # assert num_docs % 2 == 1
                world_size = self.my_world_size
                anchors = torch.arange(bsz, device=device, dtype=dtype) * num_docs + self.get_delta(bsz, num_docs)
                anchors = anchors.unsqueeze(1)  # [bsz,1]
                idxs = torch.arange(world_size * bsz * num_docs, device=device, dtype=dtype).unsqueeze(
                    0)  # .repeat(bsz, 1)
            else:
                anchors = 0
                idxs = torch.arange(num_docs, device=device, dtype=dtype).unsqueeze(0).repeat(bsz, 1)
            pos_mask = (idxs == anchors).to(dtype)
        return pos_mask

    def forward(
            self,
            input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None,
            input_ids_query=None, attention_mask_query=None, token_type_ids_query=None, position_ids_query=None,
            distill_labels=None, training_progress=None, training_mode=None,
            **kwargs,
    ):


        if training_mode is None:  # if not training, work like encoder.
            return get_representation_tensor(self.encoder(input_ids, attention_mask, )).contiguous()

        if self.model_args.tch_no_drop:
            self.sparse_encoder.eval()
            self.xe_tch_encoder.eval()

        dict_for_meta = {}

        # input reshape
        (high_dim_flag, num_docs), (org_doc_shape, org_query_shape), \
        (input_ids, attention_mask, token_type_ids, position_ids,), \
        (input_ids_query, attention_mask_query, token_type_ids_query, position_ids_query,) = preproc_inputs(
            input_ids, attention_mask, token_type_ids, position_ids,
            input_ids_query, attention_mask_query, token_type_ids_query, position_ids_query,)
        bsz = org_doc_shape[0]

        # encoding
        doc_outputs = self.encoding_doc(input_ids, attention_mask, return_dict=True)
        query_outputs = self.encoding_query(input_ids_query, attention_mask_query, return_dict=True)
        # for distill
        with torch.no_grad():
            tch_doc_outputs = self.sparse_encoding_doc(input_ids, attention_mask, return_dict=True)
            tch_query_outputs = self.sparse_encoding_query(input_ids_query, attention_mask_query, return_dict=True)
            if self.model_args.xe_tch_encoder == 'xencoder':
                cross_input_ids, cross_attention_mask, cross_token_type_ids = combine_dual_inputs_by_attention_mask(
                                input_ids_query.unsqueeze(1).expand(-1, num_docs, -1).contiguous().view(bsz*num_docs, -1),
                                attention_mask_query.unsqueeze(1).expand(-1, num_docs, -1).contiguous().view(bsz*num_docs, -1),
                                input_ids, attention_mask,
                                )
                cross_outputs = self.xe_tch_encoder(cross_input_ids, cross_attention_mask, cross_token_type_ids, output_attentions=True, output_hidden_states=True)
                xe_tch_scores = cross_outputs.logits.squeeze(-1).view(bsz, num_docs)  # [bs,num]
            else:
                logger.info("Dont't support xe_tch_encoder: {}".format(self.model_args.xe_tch_encoder))
                pass

        emb_doc, emb_query = get_representation_tensor(doc_outputs), get_representation_tensor(query_outputs)
        sp_tch_emb_doc, sp_tch_emb_query = get_representation_tensor(tch_doc_outputs), get_representation_tensor(tch_query_outputs)

        emb_dim = emb_doc.shape[-1]
        sp_tch_emb_dim = sp_tch_emb_doc.shape[-1]

        ga_emb_doc, ga_emb_query = self.gather_tensor(emb_doc), self.gather_tensor(emb_query)

        dict_for_loss = {}
        if self.model_args.with_ce_loss:
            dict_for_meta["dense_xentropy_loss_weight"] = float(self.my_world_size)

            loss_fct = nn.CrossEntropyLoss()
            target = torch.arange(ga_emb_query.shape[0], device=ga_emb_query.device, dtype=torch.long) * num_docs
            similarities = self.sim_fct(ga_emb_query, ga_emb_doc.view(-1, emb_dim))

            dict_for_loss["dense_xentropy_loss"] = loss_fct(similarities, target)


        # xencoder distillation
        self_scores = self.sim_fct.forward_qd_pair(emb_query.unsqueeze(1), emb_doc.view(bsz, num_docs, -1), )  # bs,nd
        
        temp = self.model_args.xe_tch_temp
        source = torch.log_softmax(self_scores / temp, dim=-1) + 1e-9
        target = torch.softmax(xe_tch_scores / temp, dim=-1) + 1e-9
        dict_for_loss["xe_tch_dst_kl_loss"] = nn.KLDivLoss(reduction="batchmean")(source, target) * (temp ** 2)
        dict_for_meta["xe_tch_dst_kl_loss_weight"] = self.model_args.xe_tch_dst_loss_weight


        # distillation 
        if self.model_args.apply_dst:
            if self.model_args.dst_method == 'kl':
                if self.model_args.apply_inbatch:
                    tch_ga_emb_doc, tch_ga_emb_query = self.gather_tensor(sp_tch_emb_doc), self.gather_tensor(sp_tch_emb_query)
                    target_scores = self.sim_fct(tch_ga_emb_query, tch_ga_emb_doc.view(-1, sp_tch_emb_dim)).detach()
                    self_scores = similarities
                else:
                    self_scores = self.sim_fct.forward_qd_pair(
                        emb_query.unsqueeze(1), emb_doc.view(bsz, num_docs, -1), )  # bs,nd
                    target_scores = self.sim_fct.forward_qd_pair(
                        sp_tch_emb_query.unsqueeze(1), sp_tch_emb_doc.view(bsz, num_docs, -1), ).detach()  # bs,nd
                
                if self.model_args.with_pos:
                    pos_mask = None
                else:  # not include pos sample, mask out.
                    pos_mask = self.generate_positive_mask(bsz, num_docs, self.model_args.apply_inbatch, emb_query.device)

                if pos_mask is not None:
                    self_scores = self.mask_out_logits(self_scores, pos_mask)
                    target_scores = self.mask_out_logits(target_scores, pos_mask)
                
                # calc loss
                dict_for_loss["dst_kl_loss"] = nn.KLDivLoss(reduction="batchmean")(
                    torch.log_softmax(self_scores, dim=-1), torch.softmax(target_scores, dim=-1))
                dict_for_meta["dst_kl_loss_weight"] = self.model_args.dst_loss_weight

            elif self.model_args.dst_method == 'rank': # partial order (pair-wise ranking loss)
                pair_rank_criterion = torch.nn.MarginRankingLoss(margin=0)
                # build pair-wise labels via sparse models. use only negative samples.
                bi_scores = self.sim_fct.forward_qd_pair(
                    emb_query.unsqueeze(1), emb_doc.view(bsz, num_docs, -1), )  # bs,nd
                tch_bi_scores = self.sim_fct.forward_qd_pair(
                    sp_tch_emb_query.unsqueeze(1), sp_tch_emb_doc.view(bsz, num_docs, -1), ).detach()  # bs,nd
                
                if self.model_args.with_pos:
                    _, sorted_index = torch.sort(tch_bi_scores, dim=-1, descending=True)
                else:
                    tch_neg_bi_scores = tch_bi_scores[:, 1:] # [bs, nd-1] leave out positive samples
                    _, sorted_index = torch.sort(tch_neg_bi_scores, dim=-1, descending=True)
                    # sample negative rank pairs.
                    sorted_index += 1 # shift to negative pairs. 0 is the positive.
                total_score_pairs = []
                for i in range(bsz):
                    pair_indexs = torch.LongTensor(list(combinations(sorted_index[i], 2))) # rank pairs
                    num_pair = pair_indexs.shape[0]
                    total_score_pairs.append(bi_scores[i, pair_indexs.view(-1)].view(num_pair, 2))
                
                neg_scores = torch.cat(total_score_pairs, 0) # num_neg*bsz, 2
                # print(neg_scores.shape)
                pair_loss = pair_rank_criterion(neg_scores[:, 0], neg_scores[:, 1], torch.ones_like(neg_scores[:, 0]))
                dict_for_meta["dst_rank_loss_weight"] = self.model_args.dst_loss_weight
                dict_for_loss["dst_rank_loss"] = pair_loss
            else:
                raise KeyError("dst_method should be either 'kl' or 'rank'")

        loss = 0.
        for k in dict_for_loss:
            if k + "_weight" in dict_for_meta:
                loss += dict_for_meta[k + "_weight"] * dict_for_loss[k]
            else:
                loss += dict_for_loss[k]
        dict_for_loss["loss"] = loss

        dict_for_meta.update(dict_for_loss)
        return dict_for_meta


def main():
    parser = argparse.ArgumentParser()
    # add task specific hyparam
    #
    define_hparams_training(parser)

    parser.add_argument("--data_load_type", type=str, default="disk", choices=["disk", "memory"])
    parser.add_argument("--data_dir", type=str)  # princeton-nlp/sup-simcse-bert-base-uncased
    parser.add_argument("--num_negatives", type=int, default=7)
    parser.add_argument("--num_dev", type=int, default=500)

    parser.add_argument("--custom_hn_dir", type=str, )

    parser.add_argument("--prediction_source", type=str, default="dev", choices=["dev", "test2019", "test2020"])

    parser.add_argument("--ce_score_margin", type=float, default=3.0)
    parser.add_argument("--num_negs_per_system", type=int, default=8)
    parser.add_argument("--negs_sources", type=str, default=None)
    parser.add_argument("--no_title", action="store_true")
    
    # parser.add_argument("--no_title", action="store_true")
    # parser.add_argument("--encoder", type=str, default="distilbert", )  # todo: enable

    parser.add_argument("--eval_reranking_source", type=str, default=None)

    # for hard negative sampling
    parser.add_argument("--hits_num", type=int, default=1000)

    parser.add_argument("--do_hn_gen", action="store_true")  # hard negative
    parser.add_argument("--hn_gen_num", type=int, default=1000)

    # model_param_list = add_model_hyperparameters(parser)  #  todo: enable
    model_param_list = []
    add_training_hyperparameters(parser)

    args = parser.parse_args()
    accelerator = setup_prerequisite(args)

    config, tokenizer = load_config_and_tokenizer(
        args, config_kwargs={
            # "problem_type": args.problem_type,
            # "num_labels": num_labels,
        })
    for param in model_param_list:
        setattr(config, param, getattr(args, param))

    encoder_class = AutoModel

    # prepare sparse
    if args.sp_tch_encoder == "distilbert":
        sp_tch_encoder_class = DistilBertSpladeEnocder
    elif args.sp_tch_encoder == "bert":
        sp_tch_encoder_class = BertSpladeEnocder
    elif args.sp_tch_encoder == "condenser":
        sp_tch_encoder_class = ConSpladeEnocder
    elif args.sp_tch_encoder == "roberta":
        sp_tch_encoder_class = RobertaSpladeEnocder
    else:
        raise NotImplementedError(args.sp_tch_encoder)

    xe_tch_config = AutoConfig.from_pretrained(args.xe_tch_model_path,)
    if args.xe_tch_encoder == "biencoder":
        xe_tch_encoder = AutoModel.from_pretrained(args.xe_tch_model_path, config=xe_tch_config)
    elif args.xe_tch_encoder == "xencoder":
        # de_tch_encoder = BertForSequenceClassificationWithoutPooler.from_pretrained(args.xe_tch_model_path, config=de_tch_config)
        xe_tch_encoder = AutoModelForSequenceClassification.from_pretrained(args.xe_tch_model_path, config=xe_tch_config)

    encoder = encoder_class.from_pretrained(args.model_name_or_path, config=config)
    embedding_dim = encoder.embedding_dim if hasattr(encoder, "embedding_dim") else len(tokenizer.get_vocab())
    model = encoder

    if args.do_train:
        sp_tch_config = AutoConfig.from_pretrained(args.sp_tch_model_path,)
        sp_tch_encoder = sp_tch_encoder_class.from_pretrained(args.sp_tch_model_path, config=sp_tch_config)
        learner = DenseLearnerWithSpladeDistil(args, config, tokenizer, encoder, sp_tch_encoder, xe_tch_encoder)
        with accelerator.main_process_first():
            train_dataset = DatasetMarcoPassagesRanking(
                "train", args.data_dir, args.data_load_type, args, tokenizer, add_title=(not args.no_title))
        with accelerator.main_process_first():
            dev_dataset = DatasetRerank(
                "dev", args.data_dir, "memory", args, tokenizer, num_dev=args.num_dev, add_title=(not args.no_title))
        if args.negs_sources == "official":
            train_dataset.load_official_bm25_negatives(keep_num_neg=args.num_negs_per_system, )
        elif args.negs_sources.startswith("custom"):
            negs_sources = args.negs_sources[7:]
            if negs_sources == "" or negs_sources == "all":
                all_neg_names = ["bm25-off", "co-stg1", "sp-stg1", "co-stg2", "sp-stg2"]
            else:
                all_neg_names = negs_sources.split(",")
            qid2varnegs = collections.defaultdict(collections.OrderedDict)  # aggregation
            for neg_name in all_neg_names:
                neg_filepath = os.path.join(args.custom_hn_dir, neg_name + ".pkl")
                remove_cnt = 0
                if not file_exists(neg_filepath):
                    logger.warning(f"File Not Exist! {neg_filepath}")
                else:
                    for qid, neg_pids in load_pickle(neg_filepath).items():
                        qid2varnegs[int(qid)][neg_name] = neg_pids
                        # remove conflicts; added by Kai in 05/30/2022
                        pos_pids = train_dataset.qid2pids[int(qid)]
                        for neg_pid in neg_pids.copy():
                            if neg_pid in pos_pids:
                                qid2varnegs[int(qid)][neg_name].remove(neg_pid)
                                remove_cnt += 1
                logger.info(f"Remove {remove_cnt} conflicted negative pids in {neg_name}")
            qid2negatives = dict()
            for qid, name2negs in qid2varnegs.items():
                agg_negs = []
                for negs in name2negs.values():
                    agg_negs.extend(negs[:args.num_negs_per_system])
                qid2negatives[qid] = agg_negs
            train_dataset.use_new_qid2negatives(qid2negatives, accelerator=None)

        train(
            args, train_dataset, learner, accelerator, tokenizer,
            eval_dataset=dev_dataset, eval_fn=evaluate_encoder_reranking)

    if args.do_eval or args.do_prediction or args.do_hn_gen:
        if args.do_train:
            encoder = encoder_class.from_pretrained(pretrained_model_name_or_path=args.output_dir, config=config)
        else:
            encoder = model
        encoder = accelerator.prepare(encoder)

        meta_best_str = ""
        if args.do_eval:
            with accelerator.main_process_first():
                if args.eval_reranking_source is None:
                    dev_dataset = DatasetRerank(
                        "dev", args.data_dir, "memory", args, tokenizer, num_dev=None, add_title=(not args.no_title))
                else:
                    dev_dataset = DatasetCustomRerank(
                        "dev", args.data_dir, "memory", args, tokenizer, num_dev=None, add_title=(not args.no_title),
                        filepath_dev_qid2top1000pids=args.eval_reranking_source,
                    )
            best_dev_result, best_dev_metric = evaluate_encoder_reranking(
                args, dev_dataset, encoder, accelerator, global_step=None,
                save_prediction=True, tokenizer=tokenizer, key_metric_name="MRR@10",
                similarity_metric=None, query_model=None,)
            if accelerator.is_local_main_process:
                # meta_best_str += f"best_test_result: {best_dev_result}, "
                meta_best_str += json.dumps(best_dev_metric) + os.linesep
        else:
            best_dev_result = None

        if args.do_prediction:
            best_pred_result, dev_pred_metric = evaluate_dense_retreival(
                args, args.prediction_source, encoder, accelerator, global_step=None, tb_writer=None, save_prediction=False,
                key_metric_name="MRR@10", delete_model=False, add_title=(not args.no_title), query_model=None,
                tokenizer=tokenizer, faiss_mode="gpu",
            )
            # meta_best_str += json.dumps(dev_pred_metric) + os.linesep

        if accelerator.is_local_main_process:
            with open(os.path.join(args.output_dir, "best_eval_results.txt"), "w") as fp:
                fp.write(f"{best_dev_result}, {meta_best_str}")

        if args.do_hn_gen:
            get_hard_negative_by_dense_retrieval(
                args, None, encoder, accelerator, global_step=None, tb_writer=None, save_prediction=False,
                key_metric_name="MRR@10", delete_model=False, add_title=(not args.no_title), query_model=None,
                tokenizer=tokenizer, faiss_mode="gpu",
                hits=args.hn_gen_num,
            )

if __name__ == '__main__':
    main()

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

from proj_dense.train_dense_retriever import train
from proj_sparse.modeling_splade_series import add_model_hyperparameters, DistilBertSpladeEnocder, \
    BertSpladeEnocder, ConSpladeEnocder, RobertaSpladeEnocder


from peach.enc_utils.enc_learners import LearnerMixin
from proj_dense.train_dense_retriever import DenseLearner
from transformers import AutoModel

import torch
import torch.nn as nn

from peach.common import load_pickle, file_exists
from peach.enc_utils.sim_metric import Similarity
from peach.enc_utils.general import preproc_inputs



def add_training_hyperparameters(parser):
    parser.add_argument("--apply_dst", action="store_true")
    parser.add_argument("--apply_inbatch", action="store_true")

    parser.add_argument("--remove_duplicate_pids", action="store_true")
    parser.add_argument("--use_addition_qrels", action="store_true")

    parser.add_argument("--dual_reg", action="store_true")
    parser.add_argument("--dual_reg_weight", type=float, default=1.0)

    parser.add_argument("--tch_model_path", type=str, default=None)
    parser.add_argument("--tch_encoder", type=str, default='distilbert')

    parser.add_argument("--tch_no_drop", action="store_true")

    parser.add_argument("--dst_method", type=str, default='kl', choices=['kl', 'rank', 'ranknet', 'listnet', 'margin_mse'])
    parser.add_argument("--dst_loss_weight", type=float, default=1.0)
    parser.add_argument("--rank_margin", type=float, default=0.0)

    parser.add_argument("--tch_min_confidence", type=float, default=0.0)

    parser.add_argument("--with_pos", action='store_true')

    parser.add_argument("--rank_top_ratio", type=float, default=1.0)
    parser.add_argument("--scale_loss", action="store_true")

    pass

class DenseLearnerWithSpladeDistil(DenseLearner):
    def __init__(self, model_args, config, tokenizer, encoder, sparse_encoder, sparse_query_encoder=None, query_encoder=None):
        super().__init__(model_args, config, tokenizer, encoder, query_encoder, )
        self.sparse_encoder = sparse_encoder
        self.sparse_query_encoder = sparse_query_encoder
        self.dst_method = model_args.dst_method
    
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

        emb_doc, emb_query = get_representation_tensor(doc_outputs), get_representation_tensor(query_outputs)
        tch_emb_doc, tch_emb_query = get_representation_tensor(tch_doc_outputs), get_representation_tensor(tch_query_outputs)

        emb_dim = emb_doc.shape[-1]
        tch_emb_dim = tch_emb_doc.shape[-1]

        ga_emb_doc, ga_emb_query = self.gather_tensor(emb_doc), self.gather_tensor(emb_query)

        dict_for_loss = {}
        dict_for_meta["dense_xentropy_loss_weight"] = float(self.my_world_size)

        loss_fct = nn.CrossEntropyLoss()
        target = torch.arange(ga_emb_query.shape[0], device=ga_emb_query.device, dtype=torch.long) * num_docs
        similarities = self.sim_fct(ga_emb_query, ga_emb_doc.view(-1, emb_dim))

        # dual reg.
        if self.model_args.dual_reg:  # bidirectional KL losses with same query encoder
            # forward docs again (with different dropout)
            _doc_outputs = self.encoding_doc(input_ids, attention_mask, return_dict=True)
            _emb_doc = get_representation_tensor(_doc_outputs)
            _ga_emb_doc = self.gather_tensor(_emb_doc)
            _similarities = self.sim_fct(ga_emb_query, _ga_emb_doc.view(-1, emb_dim))

            reg_loss1 = nn.KLDivLoss(reduction="batchmean")(
                    torch.log_softmax(similarities, dim=-1), torch.softmax(_similarities, dim=-1))
            reg_loss2 = nn.KLDivLoss(reduction="batchmean")(
                    torch.log_softmax(_similarities, dim=-1), torch.softmax(similarities, dim=-1))
            dict_for_loss["dual_regularization_loss"] = (reg_loss1 + reg_loss2) / 2
            dict_for_meta["dual_regularization_loss_weight"] = self.model_args.dual_reg_weight

            dict_for_loss["dense_xentropy_loss"] = 0.5 * loss_fct(similarities, target) + 0.5 * loss_fct(_similarities, target)

        else:
            dict_for_loss["dense_xentropy_loss"] = loss_fct(similarities, target)

        # distillation
        if self.model_args.apply_dst:
            if self.model_args.dst_method == 'kl':
                if self.model_args.apply_inbatch:
                    tch_ga_emb_doc, tch_ga_emb_query = self.gather_tensor(tch_emb_doc), self.gather_tensor(tch_emb_query)
                    target_scores = self.sim_fct(tch_ga_emb_query, tch_ga_emb_doc.view(-1, tch_emb_dim)).detach()
                    self_scores = similarities
                else:
                    self_scores = self.sim_fct.forward_qd_pair(
                        emb_query.unsqueeze(1), emb_doc.view(bsz, num_docs, -1), )  # bs,nd
                    target_scores = self.sim_fct.forward_qd_pair(
                        tch_emb_query.unsqueeze(1), tch_emb_doc.view(bsz, num_docs, -1), ).detach()  # bs,nd
                
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

            elif self.model_args.dst_method == 'rank':  # partial order (pair-wise ranking loss)
                pair_rank_criterion = torch.nn.MarginRankingLoss(margin=self.model_args.rank_margin)
                # build pair-wise labels via sparse models. use only negative samples.
                bi_scores = self.sim_fct.forward_qd_pair(
                    emb_query.unsqueeze(1), emb_doc.view(bsz, num_docs, -1), )  # bs,nd
                tch_bi_scores = self.sim_fct.forward_qd_pair(
                    tch_emb_query.unsqueeze(1), tch_emb_doc.view(bsz, num_docs, -1), ).detach()  # bs,nd
                
                if self.model_args.with_pos:
                    sorted_scores, sorted_index = torch.sort(tch_bi_scores, dim=-1, descending=True)
                else:
                    tch_neg_bi_scores = tch_bi_scores[:, 1:] # [bs, nd-1] leave out positive samples
                    sorted_scores, sorted_index = torch.sort(tch_neg_bi_scores, dim=-1, descending=True)
                    # sample negative rank pairs.
                    sorted_index += 1 # shift to negative pairs. 0 is the positive.
                total_score_pairs = []
                for i in range(bsz):
                    # score_range = sorted_scores[i][0] - sorted_scores[i][-1] # max - min
                    # logger.info(f"score_range: {score_range}")
                    pair_indexs = torch.LongTensor(list(combinations(sorted_index[i], 2))) # rank pairs
                    num_pair = pair_indexs.shape[0]
                    self_pairs = bi_scores[i, pair_indexs.view(-1)].view(num_pair, 2) # [num_pair, 2]
                    # save top ratio of pairs
                    tch_pairs = tch_bi_scores[i, pair_indexs.view(-1)].view(num_pair, 2)
                    logger.info(f"tch_pairs: {tch_pairs}")
                    tch_score_confidences = (tch_pairs[:, 0] - tch_pairs[:, 1])
                    valid_pair_index = tch_score_confidences >= self.model_args.tch_min_confidence
                    logger.info(f"tch_confidences: {tch_score_confidences}")
                    # logger.info("pairs: {}".format(tch_pairs))
                    # logger.info("valid_pair_index: {}".format(valid_pair_index))

                    total_score_pairs.append(self_pairs[valid_pair_index])
                
                neg_scores = torch.cat(total_score_pairs, 0) # num_neg*bsz, 2
                # print(neg_scores.shape)
                pair_loss = pair_rank_criterion(neg_scores[:, 0], neg_scores[:, 1], torch.ones_like(neg_scores[:, 0]))
                dict_for_meta["dst_rank_loss_weight"] = self.model_args.dst_loss_weight
                if self.model_args.scale_loss:  # scale up the loss w.r.t. rank top ratio
                    dict_for_loss["dst_rank_loss"] = pair_loss / self.model_args.rank_top_ratio
                else:
                    dict_for_loss["dst_rank_loss"] = pair_loss

            elif self.model_args.dst_method == 'ranknet':  # default updating teachers!
                # for score pairs of teachers,  compute entropy loss.
                bi_scores = self.sim_fct.forward_qd_pair(
                    emb_query.unsqueeze(1), emb_doc.view(bsz, num_docs, -1), )  # bs,nd
                tch_bi_scores = self.sim_fct.forward_qd_pair(
                    tch_emb_query.unsqueeze(1), tch_emb_doc.view(bsz, num_docs, -1), )  # bs,nd
                
                if self.model_args.with_pos:
                    pass
                else:
                    tch_bi_scores, bi_scores = tch_bi_scores[:, 1:], bi_scores[:, 1:] # [bs, nd-1] leave out positive samples

                score_num = tch_bi_scores.shape[1] # num of scores for each query
                
                pair_indexs = torch.LongTensor(list(combinations(range(score_num), 2)))
                num_pair = pair_indexs.shape[0]
                tch_score_pairs = tch_bi_scores[:,  pair_indexs.view(-1)].view(-1, 2)
                stu_score_pairs = bi_scores[:,  pair_indexs.view(-1)].view(-1, 2)

                # calc loss
                loss = - torch.sum((tch_score_pairs[:, 0] - tch_score_pairs[:, 1]) * torch.log(torch.sigmoid(stu_score_pairs[:, 0] - stu_score_pairs[:, 1])))
                dict_for_meta["dst_ranknet_loss_weight"] = self.model_args.dst_loss_weight
                dict_for_loss["dst_ranknet_loss"] = loss
            elif self.model_args.dst_method == 'listnet':  # default updating teachers!
                if self.model_args.apply_inbatch:
                    tch_ga_emb_doc, tch_ga_emb_query = self.gather_tensor(tch_emb_doc), self.gather_tensor(tch_emb_query)
                    target_scores = self.sim_fct(tch_ga_emb_query, tch_ga_emb_doc.view(-1, tch_emb_dim))
                    self_scores = similarities
                else:
                    self_scores = self.sim_fct.forward_qd_pair(
                        emb_query.unsqueeze(1), emb_doc.view(bsz, num_docs, -1), )  # bs,nd
                    target_scores = self.sim_fct.forward_qd_pair(
                        tch_emb_query.unsqueeze(1), tch_emb_doc.view(bsz, num_docs, -1), )  # bs,nd

                self_scores = torch.softmax(self_scores, dim=-1)
                target_scores = torch.softmax(target_scores, dim=-1)
                # calc loss
                loss = - torch.sum(target_scores * torch.log(self_scores))
                dict_for_meta["dst_listnet_loss_weight"] = self.model_args.dst_loss_weight
                dict_for_loss["dst_listnet_loss"] = loss
            elif self.model_args.dst_method == 'margin_mse':
                if self.model_args.apply_inbatch:
                    tch_ga_emb_doc, tch_ga_emb_query = self.gather_tensor(tch_emb_doc), self.gather_tensor(tch_emb_query)
                    target_scores = self.sim_fct(tch_ga_emb_query, tch_ga_emb_doc.view(-1, tch_emb_dim))  # bs,nd
                    self_scores = similarities
                else:
                    self_scores = self.sim_fct.forward_qd_pair(
                        emb_query.unsqueeze(1), emb_doc.view(bsz, num_docs, -1), )  # bs,nd
                    target_scores = self.sim_fct.forward_qd_pair(
                        tch_emb_query.unsqueeze(1), tch_emb_doc.view(bsz, num_docs, -1), )  # bs,nd
                
                self_scores = torch.softmax(self_scores, dim=-1)
                target_scores = torch.softmax(target_scores, dim=-1)
                # calc loss via metrix
                target_diffs = target_scores.unsqueeze(2) - target_scores.unsqueeze(1)
                self_diffs = self_scores.unsqueeze(2) - self_scores.unsqueeze(1)
                mean_squared_error = torch.nn.MSELoss()
                loss = mean_squared_error(target_diffs, self_diffs)
                dict_for_meta["dst_margin_mse_loss_weight"] = self.model_args.dst_loss_weight
                dict_for_loss["dst_margin_mse_loss"] = loss
            else:
                raise KeyError("dst_method should be either 'kl', 'rank', 'ranknet', 'listnet', and 'margin_mse' ")

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

    parser.add_argument("--prediction_source", type=str, default="dev", choices=["dev", "test2019", "test2020"])

    parser.add_argument("--custom_hn_dir", type=str)

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
    if args.tch_encoder == "distilbert":
        tch_encoder_class = DistilBertSpladeEnocder
    elif args.tch_encoder == "bert":
        tch_encoder_class = BertSpladeEnocder
    elif args.tch_encoder == "condenser":
        tch_encoder_class = ConSpladeEnocder
    elif args.tch_encoder == "roberta":
        tch_encoder_class = RobertaSpladeEnocder
    else:
        raise NotImplementedError(args.tch_encoder)

    encoder = encoder_class.from_pretrained(args.model_name_or_path, config=config)
    embedding_dim = encoder.embedding_dim if hasattr(encoder, "embedding_dim") else len(tokenizer.get_vocab())
    model = encoder

    if args.do_train:
        tch_config = AutoConfig.from_pretrained(args.tch_model_path,)
        tch_encoder = tch_encoder_class.from_pretrained(args.tch_model_path, config=tch_config)
        learner = DenseLearnerWithSpladeDistil(args, config, tokenizer, encoder, tch_encoder, query_encoder=None)
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
                logger.info(f"average length of {neg_name} is {np.mean([len(qid2varnegs[k][neg_name]) for k in qid2varnegs.keys()])}")
                logger.info(f"Remove {remove_cnt} conflicted negative pids in {neg_name}")

            qid2negatives = dict()
            for qid, name2negs in qid2varnegs.items():
                agg_negs = []
                for negs in name2negs.values():
                    agg_negs.extend(negs[:args.num_negs_per_system])
                if args.remove_duplicate_pids:
                    qid2negatives[qid] = list(set(agg_negs))
                else:
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

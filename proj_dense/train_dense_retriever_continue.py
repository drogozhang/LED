import os
import random
import copy
import json
import pickle

import tqdm

import torch
import torch.nn as nn
from transformers import AutoModel

from peach.base import *
from peach.common import save_jsonl, save_pickle, file_exists
from peach.datasets.marco.dataset_marco_passages import DatasetMarcoPassagesRanking
from peach.datasets.marco.dataset_marco_eval import DatasetRerank
from peach.enc_utils.eval_functions import evaluate_encoder_reranking
from peach.enc_utils.eval_dense import evaluate_dense_retreival
from peach.enc_utils.general import get_representation_tensor
from peach.enc_utils.enc_learners import LearnerMixin
from peach.enc_utils.sim_metric import Similarity
from peach.enc_utils.general import preproc_inputs
from proj_dense.train_dense_retriever import DenseLearner, train, add_training_hyperparameters
# dense util
from proj_dense.dense_utils import get_hard_negative_by_retriever


def main():
    parser = argparse.ArgumentParser()
    # add task specific hyparam
    define_hparams_training(parser)
    # add for 2 stage:
    parser.add_argument("--do_mix_negatives", action="store_true", )
    parser.add_argument("--static_hard_negatives_path", type=str, default=None)

    parser.add_argument("--data_load_type", type=str, default="disk", choices=["disk", "memory"])
    parser.add_argument("--data_dir", type=str,
                        default=USER_HOME + "/ws/data/set/")  # princeton-nlp/sup-simcse-bert-base-uncased
    parser.add_argument("--num_negatives", type=int, default=7)
    parser.add_argument("--num_dev", type=int, default=500)

    parser.add_argument("--ce_score_margin", type=float, default=3.0)
    parser.add_argument("--num_negs_per_system", type=int, default=8)
    parser.add_argument("--negs_sources", type=str, default=None)
    parser.add_argument("--no_title", action="store_true")
    # parser.add_argument("--encoder", type=str, default="distilbert", )  # todo: enable

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
    encoder = encoder_class.from_pretrained(args.model_name_or_path, config=config)
    embedding_dim = encoder.embedding_dim if hasattr(encoder, "embedding_dim") else len(tokenizer.get_vocab())
    model = encoder

    if args.do_train:
        model = DenseLearner(args, config, tokenizer, encoder, query_encoder=None)
        with accelerator.main_process_first():  # train data
            train_dataset = DatasetMarcoPassagesRanking(
                "train", args.data_dir, args.data_load_type, args, tokenizer, add_title=(not args.no_title))

        train_dataset.load_official_bm25_negatives(accelerator, keep_num_neg=args.num_negs_per_system, )
        # static_hard_neg_file = os.path.join(args.static_hard_negatives_path, "qid2negatives.pkl")
        if file_exists(args.static_hard_negatives_path):
            logger.info("Reading hard negs ...")
            with open(args.static_hard_negatives_path, "rb") as fp:
                qid2negatives = pickle.load(fp)
        else:
            raise ValueError("Static hard negative file ({}) does not exist. Check again! ".format(args.static_hard_negatives_path))
        # mix with previous negatives
        for qid in qid2negatives:
            qid2negatives[qid] = qid2negatives[qid][:args.num_negs_per_system]  # only keep top negs

        if args.do_mix_negatives and accelerator.is_local_main_process:
            assert train_dataset.qid2negatives is not None
            logger.info("Mixing BM25 negative ...")
            for qid in qid2negatives:
                try:
                    previous_negatives = copy.copy(train_dataset.qid2negatives[qid])
                except KeyError:
                    previous_negatives = []
                random.shuffle(previous_negatives)
                qid2negatives[qid].extend(previous_negatives[:args.num_negs_per_system//2])
                qid2negatives[qid] = list(set(qid2negatives[qid]))  # remove duplicates
        train_dataset.use_new_qid2negatives(qid2negatives, accelerator)

        with accelerator.main_process_first():
            dev_dataset = DatasetRerank(
                "dev", args.data_dir, "memory", args, tokenizer, num_dev=args.num_dev, add_title=(not args.no_title))

        train(args, train_dataset, model, accelerator, tokenizer, eval_dataset=dev_dataset, eval_fn=evaluate_encoder_reranking)

    if args.do_eval or args.do_prediction:
        if args.do_train:
            encoder = encoder_class.from_pretrained(pretrained_model_name_or_path=args.output_dir, config=config)
        else:
            encoder = model
        encoder = accelerator.prepare(encoder)

        meta_best_str = ""
        if args.do_eval:
            with accelerator.main_process_first():
                dev_dataset = DatasetRerank(
                    "dev", args.data_dir, "memory", args, tokenizer, num_dev=None, add_title=(not args.no_title))

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
                args, None, encoder, accelerator, global_step=None, tb_writer=None, save_prediction=False,
                key_metric_name="MRR@10", delete_model=False, add_title=(not args.no_title), query_model=None,
                tokenizer=tokenizer, faiss_mode="gpu"
            )
            # meta_best_str += json.dumps(dev_pred_metric) + os.linesep

        if accelerator.is_local_main_process:
            with open(os.path.join(args.output_dir, "best_eval_results.txt"), "w") as fp:
                fp.write(f"{best_dev_result}, {meta_best_str}")


if __name__ == '__main__':
    main()

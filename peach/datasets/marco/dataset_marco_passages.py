from asyncio.log import logger
import os

import torch
from torch.utils.data.dataset import Dataset

from peach.common import file_exists, get_line_offsets, save_json, load_json, \
    load_list_from_file, load_tsv, http_get, save_pickle, load_pickle
import json
from tqdm import tqdm
import random
from transformers import AutoTokenizer
from peach.base import CustomArgs
import collections
from copy import copy
import torch.distributed as dist
import logging
import gzip, pickle

from peach.enc_utils.general import MAX_QUERY_LENGTH


class DatasetMarcoPassagesRanking(Dataset):
    DATA_TYPE_SET = set(["train", ])
    LOAD_TYPE_SET = set(["memory", "disk"])

    # MSMARCO_PASSAGE_DEV_QRELS_FILENAME = "msmarco/passage_ranking/qrels.dev.tsv"
    MSMARCO_PASSAGE_TRAIN_QRELS_FILENAME = "msmarco/passage_ranking/qrels.train.tsv"
    MSMARCO_PASSAGE_TRAIN_QUERY_FILENAME = "msmarco/passage_ranking/train.query.txt"

    # BM25 negative
    MSMARCO_PASSAGE_OFFICIAL_NEGATIVE_FILENAME = "msmarco/passage_ranking/train.negatives.tsv"
    # Diverse negative

    # cross-enc scores
    CE_SCORES_FILENAME = 'msmarco/cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl.gz'
    SBERT_NEGATIVE_FILENAME = 'msmarco/msmarco-hard-negatives.jsonl.gz'

    MSMARCO_PASSAGE_COLLECTION_FILENAME = "msmarco/collection.tsv"

    def __init__(
            self, data_type, data_dir, load_type, data_args, tokenizer, add_title=True, **kwargs):
        assert data_type in self.DATA_TYPE_SET
        assert load_type is None or load_type in self.LOAD_TYPE_SET

        self.data_type = data_type
        self.data_dir = data_dir
        self.load_type = load_type or "disk"
        self.data_args = data_args
        self.tokenizer = tokenizer
        self.add_title = add_title

        if hasattr(self.data_args, "use_addition_qrels") and self.data_args.use_addition_qrels:
            self.MSMARCO_PASSAGE_TRAIN_QRELS_FILENAME = "msmarco/passage_ranking/qrels.train.all.tsv"

        self.num_negatives = data_args.num_negatives
        # self.tie_encoder = hasattr(self.data_args, "tie_encoder") and self.data_args.tie_encoder

        # collection pre-process
        self.collection_path = os.path.join(self.data_dir, self.MSMARCO_PASSAGE_COLLECTION_FILENAME)

        if load_type == "disk":
            pid2offset_file_path = self.collection_path + ".pid2offset.json"
            if file_exists(pid2offset_file_path):
                pid2offset = dict((int(k), v) for k, v in load_json(pid2offset_file_path).items())
            else:
                offset_file_path = self.collection_path + ".offset.json"
                if file_exists(offset_file_path):
                    offsets = load_json(offset_file_path)
                else:
                    offsets = get_line_offsets(self.collection_path, encoding="utf-8")
                    save_json(offsets, offset_file_path)
                pid2offset = dict()
                with open(self.collection_path, encoding="utf-8") as fp:

                    for idx_line, line in enumerate(fp):
                        passage_id, _ = line.strip("\n").split("\t")
                        pid2offset[int(passage_id)] = offsets[idx_line]
                    assert len(pid2offset) == len(offsets)
                    save_json(pid2offset, pid2offset_file_path)
            self.collection_pid2offset = pid2offset
            self.collection_pid_list = list(self.collection_pid2offset.keys())
        else:
            pid_text = load_tsv(self.collection_path)
            self.collection_pid2text = dict((int(pid), text) for pid, text in pid_text)

        # collection title
        if self.add_title:
            pid_title = load_tsv(self.collection_path + ".title.tsv")
            self.pid2title = dict((int(pid), title) for pid, title in pid_title)
        else:
            self.pid2title = None

        # load queries
        queries = load_tsv(os.path.join(self.data_dir, self.MSMARCO_PASSAGE_TRAIN_QUERY_FILENAME))
        self.qid2query = dict((int(qid), query_text) for qid, query_text in queries)
        assert len(queries) == len(self.qid2query)

        qrels = load_tsv(os.path.join(self.data_dir, self.MSMARCO_PASSAGE_TRAIN_QRELS_FILENAME))
        self.qid2pids = collections.defaultdict(set)
        self.qrels = []  # one query multi positive
        for qrel in qrels:
            assert len(qrel) == 4
            qid, pid = int(qrel[0]), int(qrel[2])
            self.qid2pids[qid].add(pid)
            self.qrels.append((qid, pid, ))

        # load negative samples
        self.qid2negatives = None
        self.qp2scores_distill = None  #

        self.example_list = self.qrels

    def load_official_bm25_negatives(self, accelerator=None, keep_num_neg=None):
        qid_pids_list = load_tsv(
            os.path.join(self.data_dir, self.MSMARCO_PASSAGE_OFFICIAL_NEGATIVE_FILENAME))
        qid2pids = {}
        for qid, pids in qid_pids_list:
            neg_pids = [int(pid) for pid in pids.split(",")]
            if keep_num_neg is not None:
                neg_pids = neg_pids[:keep_num_neg]
            qid2pids[int(qid)] = neg_pids
        # qid2pids = dict((int(k), v) for k, v in load_json(os.path.join(self.data_args.output_dir, "tmp_qid2negatives.json")).items())
        self.use_new_qid2negatives(qid2pids)

        if accelerator is not None and dist.is_initialized():
            accelerator.wait_for_everyone()

    def load_sbert_hard_negatives(
            self, accelerator=None, ce_score_margin=3.0, num_negs_per_system=8, negs_sources=None,
            **kwargs, ):
        # part 1: ce scores
        ce_scores_file = os.path.join(self.data_dir, self.CE_SCORES_FILENAME)
        if not file_exists(ce_scores_file):
            logging.info("Download cross-encoder scores file")
            http_get(
                'https://huggingface.co/datasets/sentence-transformers/msmarco-hard-negatives/resolve/main/cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl.gz',
                ce_scores_file)
        logging.info("Load CrossEncoder scores dict")
        with gzip.open(ce_scores_file, 'rb') as fIn:
            ce_scores = pickle.load(fIn)
        self.qp2scores_distill = ce_scores

        # print(list(self.qp2scores_distill.keys())[:10])
        # part 2: xxx
        hard_negatives_filepath = os.path.join(self.data_dir, self.SBERT_NEGATIVE_FILENAME)
        if not os.path.exists(hard_negatives_filepath):
            logging.info("Download cross-encoder scores file")
            http_get(
                'https://huggingface.co/datasets/sentence-transformers/msmarco-hard-negatives/resolve/main/msmarco-hard-negatives.jsonl.gz',
                hard_negatives_filepath)
        logging.info("Read hard negatives train file")

        qid2pids = {}
        negs_to_use = None
        with gzip.open(hard_negatives_filepath, 'rt') as fIn:
            for line in tqdm(fIn):
                data = json.loads(line)

                # Get the positive passage ids
                qid = data['qid']
                pos_pids = data['pos']

                if len(pos_pids) == 0:  # Skip entries without positives passages
                    continue

                pos_min_ce_score = min([ce_scores[qid][pid] for pid in data['pos']])
                ce_score_threshold = pos_min_ce_score - ce_score_margin

                # Get the hard negatives
                neg_pids = set()
                if negs_to_use is None:
                    if negs_sources is not None:  # Use specific system for negatives
                        negs_to_use = negs_sources.split(",")
                    else:  # Use all systems
                        negs_to_use = list(data['neg'].keys())
                    logging.info("Using negatives from the following systems:{}".format(negs_to_use))

                for system_name in negs_to_use:
                    if system_name not in data['neg']:
                        continue

                    system_negs = data['neg'][system_name]
                    negs_added = 0
                    for pid in system_negs:
                        if ce_scores[qid][pid] > ce_score_threshold:
                            continue

                        if pid not in neg_pids:
                            neg_pids.add(pid)
                            negs_added += 1
                            if negs_added >= num_negs_per_system:
                                break
                if len(neg_pids) > 0:
                    qid2pids[qid] = list(neg_pids)

                # if args.use_all_queries or (len(pos_pids) > 0 and len(neg_pids) > 0):
                #     train_queries[data['qid']] = {'qid': data['qid'], 'query': queries[data['qid']], 'pos': pos_pids,
                #                                   'neg': neg_pids}
        # logging.info("Train queries: {}".format(len(train_queries)))
        self.use_new_qid2negatives(qid2pids)
        if accelerator is not None and dist.is_initialized():
            accelerator.wait_for_everyone()

    def use_new_qid2negatives(self, qid2negatives, accelerator=None):
        # need `accelerator` for multi-processes
        if accelerator is None or (not dist.is_initialized()):
            self.qid2negatives = qid2negatives
        else:
            # use file to sync all processes
            tmp_path = os.path.join(self.data_args.output_dir, "tmp_qid2negatives.pkl")
            if accelerator.is_local_main_process:
                save_pickle(qid2negatives, tmp_path)
            accelerator.wait_for_everyone()
            self.qid2negatives = dict((int(k), v) for k, v in load_pickle(tmp_path).items())
            accelerator.wait_for_everyone()
        self.example_list = [
            (qid, pid, ) for (qid, pid, ) in self.qrels if qid in self.qid2negatives]

    def __len__(self):
        return len(self.example_list)

    def sample_negatives(self, negatives, num_negatives):
        try:
            if len(negatives) == num_negatives:
                neg_pids = negatives
            elif len(negatives) < num_negatives:
                # neg_pids = negatives + random.choices(self.collection_pid_list, k=self.num_negatives-len(negatives))
                neg_pids = [negatives[i % len(negatives)] for i in range(num_negatives)]
            else:
                negatives_copy = copy(negatives)
                random.shuffle(negatives_copy)
                neg_pids = negatives_copy[:num_negatives]
        except KeyError:
            neg_pids = random.choices(self.collection_pid_list, k=num_negatives)
        return neg_pids

    def __getitem__(self, item):
        qid, pos_pid = self.example_list[item]

        # true_positives = self.qid2pids[qid]
        # negative sampling
        if self.qid2negatives is None:
            neg_pids = random.choices(self.collection_pid_list, k=self.num_negatives)
        else:
            negatives = self.qid2negatives[qid]
            if len(negatives) > 0 and isinstance(negatives[0], tuple) and isinstance(negatives[0][0], str):
                assert self.num_negatives % len(negatives) == 0
                neg_pids = []
                for neg_name, negs in negatives:
                    neg_pids.extend(self.sample_negatives(negs, self.num_negatives//len(negatives)))
            else:
                neg_pids = self.sample_negatives(negatives, self.num_negatives)

        # read dataset
        query = self.qid2query[qid]

        all_passages = []
        with open(self.collection_path, encoding="utf-8") as fp:
            for pid in [pos_pid, ] + neg_pids:
                if self.load_type == "disk":
                    fp.seek(self.collection_pid2offset[pid])
                    line = fp.readline()
                    passage_id, passage = line.strip("\n").split("\t")
                    assert int(passage_id) == pid
                    all_passages.append(passage)
                else:
                    all_passages.append(self.collection_pid2text[pid])
        all_text = all_passages

        if self.add_title and self.pid2title is not None:
            all_titles = []
            for pid in [pos_pid, ] + neg_pids:
                all_titles.append(self.pid2title[pid])
            # all_text = [self.tokenizer.sep_token.join([t, p, ]) for t, p in zip(all_titles, all_passages, )]
            all_text = [(t, p) for t, p in zip(all_titles, all_passages, )]

        # distill_labels = None
        if self.qp2scores_distill is not None:
            distill_labels = [self.qp2scores_distill[qid][pid] for pid in [pos_pid, ] + neg_pids]
        else:
            distill_labels = None

        query_outputs = self.tokenizer(
            query,
            add_special_tokens=True,
            max_length=MAX_QUERY_LENGTH, truncation=True)
        if "token_type_ids" in query_outputs:
            query_outputs["token_type_ids"] = [1] * len(query_outputs["token_type_ids"])  # tbd

        passage_outputs = self.tokenizer(
            *zip(*all_text),
            add_special_tokens=True,
            max_length=self.data_args.max_length, truncation=True)  # "only_second"

        # feature_dict = {
        #         "input_ids": [query_outputs["input_ids"], ] + passage_outputs["input_ids"],
        #         "attention_mask": [query_outputs["attention_mask"], ] + passage_outputs["attention_mask"],}
        # if "token_type_ids" in passage_outputs:
        #     feature_dict["token_type_ids"] = [query_outputs["token_type_ids"], ] + passage_outputs["token_type_ids"]

        feature_dict = {
                "input_ids": passage_outputs["input_ids"],
                "attention_mask": passage_outputs["attention_mask"],
                "input_ids_query": query_outputs["input_ids"],
                "attention_mask_query": query_outputs["attention_mask"],
            }
        # if "token_type_ids" in passage_outputs:
        #     feature_dict["token_type_ids"] = passage_outputs["token_type_ids"]
        #     feature_dict["token_type_ids_query"] = query_outputs["token_type_ids"]

        if distill_labels is not None:
            feature_dict["distill_labels"] = distill_labels

        return feature_dict






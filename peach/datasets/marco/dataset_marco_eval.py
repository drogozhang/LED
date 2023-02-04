import os
from torch.utils.data.dataset import Dataset
from peach.common import file_exists, get_line_offsets, save_json, load_json, load_tsv, load_pickle
from transformers import AutoTokenizer
import collections
from peach.enc_utils.general import MAX_QUERY_LENGTH


class DatasetMacroPassages(Dataset):
    DATA_TYPE_SET = set(["train", "dev", "test2019", "test2020"])

    MSMARCO_PASSAGE_COLLECTION_FILENAME = "msmarco/collection.tsv"

    def __init__(self, data_type, data_dir, data_names, data_args, tokenizer, add_title=True):
        assert data_type in self.DATA_TYPE_SET

        self.data_type = data_type
        self.data_dir = data_dir
        self.data_args = data_args
        self.tokenizer = tokenizer
        self.add_title = add_title

        self.data_path = os.path.join(self.data_dir, self.MSMARCO_PASSAGE_COLLECTION_FILENAME)

        offset_file_path = self.data_path + ".offset.json"
        if file_exists(offset_file_path):
            offsets = load_json(offset_file_path)
        else:
            offsets = get_line_offsets(self.data_path, encoding="utf-8")
            save_json(offsets, offset_file_path)
        self.example_list = offsets

        # add title
        if self.add_title:
            pid_title = load_tsv(self.data_path + ".title.tsv")
            self.pid2title = dict((int(pid), title) for pid, title in pid_title)
        else:
            self.pid2title = None

    def __len__(self):
        return len(self.example_list)

    def __getitem__(self, item):
        with open(self.data_path) as fp:
            fp.seek(self.example_list[item])
            raw_text = fp.readline()
        passage_id, passage = raw_text.strip("\n").split("\t")
        passage_id = int(passage_id)

        if self.add_title:
            title_plus_passage = self.tokenizer.sep_token.join([self.pid2title[passage_id], passage,])
            passage_outputs = self.tokenizer(
                title_plus_passage,
                add_special_tokens=True,
                # return_offsets_mapping=True,
                max_length=self.data_args.max_length, truncation=True)  # "only_second"
        else:
            passage_outputs = self.tokenizer(
                passage,
                add_special_tokens=True,
                # return_offsets_mapping=True,
                max_length=self.data_args.max_length, truncation=True)


        # cls_id = self.tokenizer.cls_token_id
        # sep_id = self.tokenizer.sep_token_id

        return {
            "pids": passage_id,
            "input_ids": passage_outputs["input_ids"],
            "attention_mask": passage_outputs["attention_mask"],
            # "token_type_ids": passage_outputs["token_type_ids"],
        }


class DatasetFullRankQueries(Dataset):
    DATA_TYPE_SET = set(["train", "dev", "test2019", "test2020"])

    MSMARCO_PASSAGE_DEV_QRELS_FILENAME = "msmarco/passage_ranking/qrels.{}.tsv"
    MSMARCO_PASSAGE_DEV_QUERY_FILENAME = "msmarco/passage_ranking/{}.query.txt"
    # MSMARCO_PASSAGE_COLLECTION_FILENAME = "msmarco/collection.tsv"

    def __init__(self, data_type, data_dir, data_names, data_args, tokenizer, num_dev=None):
        assert data_type in self.DATA_TYPE_SET

        self.data_type = data_type
        self.data_dir = data_dir
        self.data_args = data_args
        self.tokenizer = tokenizer

        # q relevance
        if self.data_type in ["test2019", "test2020"] :
            qrels = load_tsv(os.path.join(self.data_dir, self.MSMARCO_PASSAGE_DEV_QRELS_FILENAME.format(self.data_type)))
            self.qid2pids = collections.defaultdict(set)
            for qrel in qrels:
                assert len(qrel) == 4
                if int(qrel[3]) > 0:
                    self.qid2pids[int(qrel[0])].add(int(qrel[2]))
        else:
            qrels = load_tsv(os.path.join(self.data_dir, self.MSMARCO_PASSAGE_DEV_QRELS_FILENAME.format(self.data_type)))
            self.qid2pids = collections.defaultdict(set)
            for qrel in qrels:
                assert len(qrel) == 4
                self.qid2pids[int(qrel[0])].add(int(qrel[2]))

        # dev query
        queries = load_tsv(os.path.join(self.data_dir, self.MSMARCO_PASSAGE_DEV_QUERY_FILENAME.format(self.data_type)))
        self.example_list = [(int(qid), query_text) for qid, query_text in queries]

    def __len__(self):
        return len(self.example_list)

    def __getitem__(self, item):
        qid, query = self.example_list[item]

        query_outputs = self.tokenizer(
            query,
            add_special_tokens=True,
            # return_offsets_mapping=True,
            max_length=MAX_QUERY_LENGTH, truncation=True)
        # query_outputs["token_type_ids"] = [1] * len(query_outputs["token_type_ids"])

        return {
            "qids": qid,
            "input_ids_query": query_outputs["input_ids"],
            "attention_mask_query": query_outputs["attention_mask"],
            # "token_type_ids_query": query_outputs["token_type_ids"],
        }


class DatasetCustomRerank(Dataset):
    DATA_TYPE_SET = set(["dev", "test2019", "test2020"])

    MSMARCO_PASSAGE_DEV_QUERY_FILENAME = "msmarco/passage_ranking/{}.query.txt"
    MSMARCO_PASSAGE_DEV_QRELS_FILENAME = "msmarco/passage_ranking/qrels.{}.tsv"
    MSMARCO_PASSAGE_COLLECTION_FILENAME = "msmarco/collection.tsv"

    # default
    MSMARCO_PASSAGE_DEV_TOP1000_FILENAME = "msmarco/passage_ranking/top1000.{}"

    def __init__(
            self, data_type, data_dir, load_type, data_args, tokenizer, num_dev=None, add_title=True,
            filepath_dev_qid2top1000pids=None,
    ):
        assert data_type in self.DATA_TYPE_SET

        self.data_type = data_type
        self.data_dir = data_dir
        self.load_type = load_type
        self.data_args = data_args
        self.tokenizer = tokenizer
        self.add_title = add_title
        # assert filepath_dev_qid2top1000pids is not None

        # msmarco_passage_dev_top1000_path = os.path.join(self.data_dir, self.MSMARCO_PASSAGE_DEV_TOP1000_FILENAME)
        self.collection_path = os.path.join(self.data_dir, self.MSMARCO_PASSAGE_COLLECTION_FILENAME)
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
        # self.collection_pid_list = list(self.collection_pid2offset.keys())

        if self.add_title:
            pid_title = load_tsv(os.path.join(self.data_dir, self.MSMARCO_PASSAGE_COLLECTION_FILENAME+".title.tsv"))
            self.pid2title = dict((int(pid), title) for pid, title in pid_title)
        else:
            self.pid2title = None

        # q relevance
        if self.data_type in ["test2019", "test2020"] :
            qrels = load_tsv(
                os.path.join(self.data_dir, self.MSMARCO_PASSAGE_DEV_QRELS_FILENAME.format(self.data_type)))
            self.qid2pids = collections.defaultdict(set)
            for qrel in qrels:
                assert len(qrel) == 4
                if int(qrel[3]) > 0:
                    self.qid2pids[int(qrel[0])].add(int(qrel[2]))
        else:
            qrels = load_tsv(os.path.join(self.data_dir, self.MSMARCO_PASSAGE_DEV_QRELS_FILENAME.format(self.data_type)))
            self.qid2pids = collections.defaultdict(set)
            for qrel in qrels:
                assert len(qrel) == 4
                self.qid2pids[int(qrel[0])].add(int(qrel[2]))

        # dev query
        queries = load_tsv(os.path.join(self.data_dir, self.MSMARCO_PASSAGE_DEV_QUERY_FILENAME.format(self.data_type)))
        self.qid2query = dict((int(qid), query_text) for qid, query_text in queries)

        # last get filepath_dev_qid2top1000pids
        if filepath_dev_qid2top1000pids is None or filepath_dev_qid2top1000pids == "none":
            # official bm25
            msmarco_passage_dev_top1000_path = os.path.join(self.data_dir, self.MSMARCO_PASSAGE_DEV_TOP1000_FILENAME.format(self.data_type))
            qid2top1000pids_file_path = msmarco_passage_dev_top1000_path + ".qid2top1000pids.json"
            if file_exists(qid2top1000pids_file_path):
                qid2top1000pids_str = load_json(qid2top1000pids_file_path)
                # transfer str-key to int-key
                qid2top1000pids = dict((int(k), v) for k, v in qid2top1000pids_str.items())
            else:
                qid2top1000pids = collections.defaultdict(list)
                qd_pairs = load_tsv(msmarco_passage_dev_top1000_path)
                # offsets = get_line_offsets(msmarco_passage_dev_top1000_path, encoding="utf-8")
                for idx, qd_pair in enumerate(qd_pairs):  # grouped by "qid"
                    assert len(qd_pair) == 4
                    qid, pid = int(qd_pair[0]), int(qd_pair[1])
                    qid2top1000pids[qid].append(pid)
                save_json(qid2top1000pids, qid2top1000pids_file_path)
        else:
            if filepath_dev_qid2top1000pids.endswith("pkl"):
                qid2top1000pids = load_pickle(filepath_dev_qid2top1000pids)
            elif filepath_dev_qid2top1000pids.endswith("json"):
                qid2top1000pids = load_json(filepath_dev_qid2top1000pids)
            elif filepath_dev_qid2top1000pids.endswith("trec"):
                # load trec files
                qid2top1000pids = dict()
                with open(filepath_dev_qid2top1000pids) as fp:
                    for line in fp:
                        qid, _, pid, rank, score, _method = line.strip().split(" ")
                        qid, pid, rank, score = int(qid), int(pid), int(rank), float(score)
                        if qid not in qid2top1000pids:
                            qid2top1000pids[qid] = []
                        qid2top1000pids[qid].append(pid)
            else:
                raise NotImplementedError(filepath_dev_qid2top1000pids)
        qid2top1000pids = dict((int(qid), top1000pids) for qid, top1000pids in qid2top1000pids.items())

        self.qid_list = list(sorted(self.qid2query.keys()))[:num_dev]

        self.example_list = []
        for qid in self.qid_list:
            pos_pid_set = self.qid2pids[qid]
            top1000pids = qid2top1000pids[qid]
            for pid in top1000pids:
                label = int(pid in pos_pid_set)
                self.example_list.append((qid, pid, label))

    def __len__(self):
        return len(self.example_list)

    def __getitem__(self, item):
        qid, pid, label = self.example_list[item]
        # read dataset
        query = self.qid2query[qid]

        with open(self.collection_path, encoding="utf-8") as fp:
            fp.seek(self.collection_pid2offset[pid])
            line = fp.readline()
            passage_id, passage = line.strip("\n").split("\t")
            assert int(passage_id) == pid

        if self.add_title:
            # title_plus_passage = self.tokenizer.sep_token.join([self.pid2title[pid], example["passage"],])
            passage_outputs = self.tokenizer(
                self.pid2title[pid], passage,
                add_special_tokens=True,
                # return_offsets_mapping=True,
                max_length=self.data_args.max_length, truncation=True)  # "only_second"
        else:
            passage_outputs = self.tokenizer(
                passage,
                add_special_tokens=True,
                # return_offsets_mapping=True,
                max_length=self.data_args.max_length, truncation=True)

        query_outputs = self.tokenizer(
            query,
            add_special_tokens=True,
            # return_offsets_mapping=True,
            max_length=MAX_QUERY_LENGTH, truncation=True)
        if "token_type_ids" in query_outputs:
            query_outputs["token_type_ids"] = [1] * len(query_outputs["token_type_ids"])

        feature_dict = {
            "input_ids": passage_outputs["input_ids"],
            "attention_mask": passage_outputs["attention_mask"],

            "input_ids_query": query_outputs["input_ids"],
            "attention_mask_query": query_outputs["attention_mask"],

            "qids": qid,
            "pids": pid,
            "binary_labels": label,
        }
        return feature_dict


class DatasetRerank(Dataset):
    DATA_TYPE_SET = set(["dev", "test2019", "test2020", ])

    MSMARCO_PASSAGE_DEV_QRELS_FILENAME = "msmarco/passage_ranking/qrels.{}.tsv"
    MSMARCO_PASSAGE_DEV_TOP1000_FILENAME = "msmarco/passage_ranking/top1000.{}"
    MSMARCO_PASSAGE_COLLECTION_FILENAME = "msmarco/collection.tsv"

    def __init__(self, data_type, data_dir, load_type, data_args, tokenizer, num_dev=None, add_title=True):
        assert data_type in self.DATA_TYPE_SET

        self.data_type = data_type
        self.data_dir = data_dir
        self.load_type = load_type
        self.data_args = data_args
        self.tokenizer = tokenizer
        self.add_title = add_title

        msmarco_passage_dev_top1000_path = os.path.join(self.data_dir, self.MSMARCO_PASSAGE_DEV_TOP1000_FILENAME.format(self.data_type))
        offset_file_path = msmarco_passage_dev_top1000_path + ".qid2offsets.json"
        if file_exists(offset_file_path):
            qid2offsets_str = load_json(offset_file_path)
            # transfer str-key to int-key
            qid2offsets = dict((int(k), v) for k, v in qid2offsets_str.items())
        else:
            qid2offsets = collections.defaultdict(list)
            qd_pairs = load_tsv(os.path.join(self.data_dir, self.MSMARCO_PASSAGE_DEV_TOP1000_FILENAME.format(self.data_type)))
            offsets = get_line_offsets(msmarco_passage_dev_top1000_path, encoding="utf-8")
            for idx, (qd_pair, offset) in enumerate(zip(qd_pairs, offsets)):  # grouped by "qid"
                assert len(qd_pair) == 4
                qid, pid = int(qd_pair[0]), int(qd_pair[1])
                qid2offsets[qid].append(offset)
            save_json(qid2offsets, offset_file_path)
        self.data_path = msmarco_passage_dev_top1000_path
        self.qid2offsets = qid2offsets

        # filter
        qid_list = list(sorted(qid2offsets.keys()))
        # if num_dev is not None:
        #     delta = min(max(1, len(qid_list) // num_dev, ), len(qid_list))
        #     qid_list = [qid for idx, qid in enumerate(qid_list) if idx % delta == 0]
        self.qid_list = qid_list[:num_dev]

        if self.data_type in ["test2019", "test2020"]:
            qrels = load_tsv(
                os.path.join(self.data_dir, self.MSMARCO_PASSAGE_DEV_QRELS_FILENAME.format(self.data_type)))
            self.qid2pids = collections.defaultdict(set)
            for qrel in qrels:
                assert len(qrel) == 4
                if int(qrel[3]) > 0:
                    self.qid2pids[int(qrel[0])].add(int(qrel[2]))
        else:
            qrels = load_tsv(os.path.join(self.data_dir, self.MSMARCO_PASSAGE_DEV_QRELS_FILENAME.format(self.data_type)))
            self.qid2pids = collections.defaultdict(set)
            for qrel in qrels:
                assert len(qrel) == 4
                self.qid2pids[int(qrel[0])].add(int(qrel[2]))

        # add title
        if self.add_title:
            pid_title = load_tsv(os.path.join(self.data_dir, self.MSMARCO_PASSAGE_COLLECTION_FILENAME+".title.tsv"))
            self.pid2title = dict((int(pid), title) for pid, title in pid_title)
        else:
            self.pid2title = None

        self.example_list = []
        for qid in self.qid_list:
            self.example_list.extend(qid2offsets[qid])

    def __len__(self):
        return len(self.example_list)

    def __getitem__(self, item):
        with open(self.data_path, encoding="utf-8") as fp:
            fp.seek(self.example_list[item])
            raw_text = fp.readline()
        qd_pair = raw_text.strip("\n").split("\t")
        assert len(qd_pair) == 4
        qid, pid = int(qd_pair[0]), int(qd_pair[1])
        example = {
            "qid": qid,
            "pid": pid,
            "query": qd_pair[2],
            "passage": qd_pair[3],
            "binary_labels": int(pid in self.qid2pids[qid]),
        }

        if self.add_title:
            # title_plus_passage = self.tokenizer.sep_token.join([self.pid2title[pid], example["passage"],])
            passage_outputs = self.tokenizer(
                self.pid2title[pid], example["passage"],
                add_special_tokens=True,
                # return_offsets_mapping=True,
                max_length=self.data_args.max_length, truncation=True)  # "only_second"
        else:
            passage_outputs = self.tokenizer(
                example["passage"],
                add_special_tokens=True,
                # return_offsets_mapping=True,
                max_length=self.data_args.max_length, truncation=True)

        query_outputs = self.tokenizer(
            example["query"],
            add_special_tokens=True,
            # return_offsets_mapping=True,
            max_length=MAX_QUERY_LENGTH, truncation=True)
        if "token_type_ids" in query_outputs:
            query_outputs["token_type_ids"] = [1] * len(query_outputs["token_type_ids"])

        # cls_id = self.tokenizer.cls_token_id
        # sep_id = self.tokenizer.sep_token_id
        feature_dict = {
            "input_ids": passage_outputs["input_ids"],
            "attention_mask": passage_outputs["attention_mask"],

            "input_ids_query": query_outputs["input_ids"],
            "attention_mask_query": query_outputs["attention_mask"],

            "qids": example["qid"],
            "pids": example["pid"],
            "binary_labels": example["binary_labels"],
        }

        # if "token_type_ids" in passage_outputs:
        #     feature_dict["token_type_ids"] = passage_outputs["token_type_ids"]
        #     feature_dict["token_type_ids_query"] = query_outputs["token_type_ids"]
        return feature_dict

if __name__ == '__main__':
    from peach.base import CustomArgs
    from tqdm import tqdm

    data_args = CustomArgs(
        max_length=128,
        # encoder_type="bi-encoder",
        # debug=False,
        per_device_eval_batch_size=32,
        num_proc=16
    )
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    from peach.base import setup_eval_dataloader

    # dev_dataset_off= DatasetRerank(
    #     "dev", "/data/shentao/corpus", None, data_args, tokenizer, num_dev=None
    # )

    dev_dataset_cst = DatasetCustomRerank(
        "dev", "/data/shentao/corpus", None, data_args, tokenizer, num_dev=None
    )

    anchor = 0


    # dataloader = setup_eval_dataloader(data_args, dev_dataset, None, use_accelerator=False)
    # for i, batch in tqdm(enumerate(dataloader)):
    #     pass
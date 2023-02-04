import json
import numpy as np
import time
import torch

from peach.datasets.marco.dataset_marco_eval import DatasetMacroPassages, DatasetFullRankQueries
from peach.base import *
from peach.common import file_exists, dir_exists, get_data_path_list, load_list_from_file, save_list_to_file
from tqdm import tqdm
import os
import pickle
import collections
from peach.common import load_tsv
from peach.enc_utils.general import get_representation_tensor
from multiprocessing import Pool
from peach.enc_utils.eval_sparse import sparse_vector_to_dict, dict_sparse_to_string, process_one_vec_example

def get_hard_negative_by_sparse_retrieval(
    args, eval_dataset, enc_model, accelerator, global_step=None, tb_writer=None, save_prediction=False,
    key_metric_name="recall@100", delete_model=False, add_title=True, query_model=None,
    vocab_id2token=None, tokenizer=None, quantization_factor=100,
    anserini_path = "ANSERINI_HOME",  # 0.14.0
    get_emb_lambda=None, hits=1000,
    **kwargs,
):
    enc_model.eval()
    if query_model is None:
        query_enc_model = enc_model
    else:
        query_model.eval()
        query_enc_model = query_model

    get_emb_lambda = get_representation_tensor if get_emb_lambda is None else get_emb_lambda

    vocab_id2token = vocab_id2token or dict((i, s) for s, i in tokenizer.get_vocab().items())

    abs_output_dir = os.path.abspath(args.output_dir)
    work_dir = os.path.join(abs_output_dir, "sparse_retrieval")

    saved_doc_emb_dir = os.path.join(work_dir, "sparse_emb_dir")
    saved_doc_emb_path_template = os.path.join(saved_doc_emb_dir, "sparse_emb_{}.jsonl")
    num_index_splits = 24

    if accelerator.is_local_main_process:
        if not dir_exists(work_dir):
            os.mkdir(work_dir)
        if not dir_exists(saved_doc_emb_dir):
            os.mkdir(saved_doc_emb_dir)
    accelerator.wait_for_everyone()
    saved_index_dir = os.path.join(work_dir, "collection_index")

    # save passage embedding into `saved_doc_emb_dir` if not indexed
    if len(get_data_path_list(saved_doc_emb_dir, suffix=".jsonl")) == 0 and (not dir_exists(saved_index_dir)):
        logging.info("Getting passage vectors ...")
        with accelerator.main_process_first():
            passages_dataset = DatasetMacroPassages(
                "dev", args.data_dir, None, args, tokenizer, add_title=add_title)
        passages_dataloader = setup_eval_dataloader(args, passages_dataset, accelerator, use_accelerator=True)
        emb_data_list = []
        for batch_idx, batch in tqdm(
                enumerate(passages_dataloader), disable=not accelerator.is_local_main_process,
                total=len(passages_dataloader), desc=f"Getting passage vectors ..."):
            with torch.no_grad():
                embs = get_emb_lambda(enc_model(**batch)).contiguous()
            pids = batch["pids"]

            pids, embs = accelerator.gather(pids), accelerator.gather(embs)

            if accelerator.is_local_main_process:
                for local_idx in range(pids.shape[0]):
                    pid = int(pids[local_idx].cpu().item())
                    emb = embs[local_idx].detach()
                    sparse_idxs = torch.nonzero(emb).squeeze(-1)
                    sparse_values = emb[sparse_idxs]
                    emb_data_list.append((len(emb_data_list), pid,
                                          (sparse_idxs.cpu().numpy(), sparse_values.cpu().numpy()),
                                          vocab_id2token, quantization_factor, tokenizer.unk_token,
                                          # for compatible w/ Pool
                                          ))
        accelerator.wait_for_everyone()
        if accelerator.is_local_main_process:
            logging.info("saving emb_data_list to files ...")
            collection_index_fp_list = [
                open(saved_doc_emb_path_template.format(str(i)), "w") for i in range(num_index_splits)]
            emb_data_list = emb_data_list[:len(passages_dataset)]  # fix pid duplicate in DDP

            pbar = tqdm(emb_data_list)
            with Pool() as p:
                for idx_p, dump_str in p.imap(process_one_vec_example, pbar, 256):
                    collection_index_fp_list[idx_p % num_index_splits].write(dump_str)

            for collection_index_fp in collection_index_fp_list:
                collection_index_fp.close()
        accelerator.wait_for_everyone()

    # query embeddings
    saved_query_emb_path = os.path.join(work_dir, "train_query_sparse_emb.tsv")
    if not file_exists(saved_query_emb_path):
        logging.info("Getting query vectors ...")
        with accelerator.main_process_first():
            queries_dataset = DatasetFullRankQueries("train", args.data_dir, None, args, tokenizer, )
        queries_dataloader = setup_eval_dataloader(args, queries_dataset, accelerator, use_accelerator=True)

        # if accelerator.is_local_main_process:
        #     saved_query_emb_fp = open(saved_query_emb_path, "w")
        emb_data_list = []
        for batch_idx, batch in tqdm(
                enumerate(queries_dataloader), disable=not accelerator.is_local_main_process,
                total=len(queries_dataloader), desc=f"Getting query vectors ..."):

            for k in list(batch.keys()):
                if k.endswith("_query"):
                    batch[k[:-6]] = batch.pop(k)
            with torch.no_grad():
                embs = get_emb_lambda(query_enc_model(**batch)).contiguous()
            qids = batch["qids"]

            qids, embs = accelerator.gather(qids), accelerator.gather(embs)

            if accelerator.is_local_main_process:
                for local_idx in range(qids.shape[0]):
                    qid = int(qids[local_idx].cpu().item())
                    emb = embs[local_idx].detach()
                    sparse_idxs = torch.nonzero(emb).squeeze(-1)
                    sparse_values = emb[sparse_idxs]
                    emb_data_list.append((len(emb_data_list), qid,
                                          (sparse_idxs.cpu().numpy(), sparse_values.cpu().numpy()),))
        accelerator.wait_for_everyone()
        if accelerator.is_local_main_process:
            logging.info("saving emb_data_list to file ...")
            emb_data_list = emb_data_list[:len(queries_dataset)]  # fix pid duplicate in DDP
            with open(saved_query_emb_path, "w") as fp:
                for emb_data in tqdm(emb_data_list):
                    idx, qid, tuple_vec = emb_data
                    dict_sparse = sparse_vector_to_dict(
                        tuple_vec, vocab_id2token, quantization_factor, dummy_token=tokenizer.unk_token)
                    fp.write(str(qid) + "\t" + dict_sparse_to_string(dict_sparse) + os.linesep)
        accelerator.wait_for_everyone()

    enc_model.train()
    if query_model is not None:
        query_model.train()

    if accelerator.is_local_main_process:
        # indexing passage vec jsonl from files
        if not dir_exists(saved_index_dir):
            logging.info("indexing via anserini ...")
            os.system(
                f"sh {anserini_path}/target/appassembler/bin/IndexCollection -collection JsonVectorCollection \
                         -input {saved_doc_emb_dir} \
                         -index {saved_index_dir} \
                         -generator DefaultLuceneDocumentGenerator -impact -pretokenized \
                         -threads {num_index_splits}")
            # -stemmer none
        # search
        search_result_path = os.path.join(work_dir, "train_search_result.trec")
        if not file_exists(search_result_path):
            logging.info("Doing search ...")
            logging.info(f"\tSpliting saved_query_emb file to file ...")
            query_emb_lines = load_list_from_file(saved_query_emb_path)
            each_split_len = 40000
            num_train_splits = math.ceil(len(query_emb_lines)/each_split_len)

            tmp_split_saved_query_emb_path = saved_query_emb_path + ".split-{}.tsv"
            tmp_split_search_result_path = search_result_path + ".split-{}.trec"
            for idx_split in range(num_train_splits):
                save_list_to_file(
                    query_emb_lines[idx_split*each_split_len: (idx_split+1)*each_split_len],
                    tmp_split_saved_query_emb_path.format(idx_split))

            logging.info(f"\tSearch for each split ...")
            for idx_split in range(num_train_splits):
                logging.info(f"\t\tFor split {idx_split} ...")
                os.system(
                    f"sh {anserini_path}/target/appassembler/bin/SearchCollection -hits {hits} -parallelism 80 \
                                 -index {saved_index_dir} \
                                 -topicreader TsvInt -topics {tmp_split_saved_query_emb_path.format(idx_split)} \
                                 -output {tmp_split_search_result_path.format(idx_split)} -format trec \
                                 -impact -pretokenized"
                )
            all_trec_lines = []
            for idx_split in range(num_train_splits):
                all_trec_lines.extend(load_list_from_file(tmp_split_search_result_path.format(idx_split)))
            save_list_to_file(all_trec_lines, search_result_path)

        # transform trec to qid2negatives
        qrels = load_tsv(os.path.join(args.data_dir, "msmarco/passage_ranking/qrels.train.tsv"))
        qid2pos_pids = collections.defaultdict(set)
        for qrel in qrels:
            assert len(qrel) == 4
            qid, pid = int(qrel[0]), int(qrel[2])
            qid2pos_pids[qid].add(pid)

        qid2all_pids = collections.defaultdict(list)
        with open(search_result_path) as search_result_fp:
            for line in search_result_fp:
                line_info = line.strip().split(" ")
                assert len(line_info) == 6
                qid, _, pid, rank, score, _method = line_info
                qid, pid, = int(qid), int(pid)
                qid2all_pids[qid].append(pid)

        qid2negatives = dict()
        for qid in qid2all_pids:
            pos_pids = qid2pos_pids[qid]
            negatives = []
            for pid in qid2all_pids[qid]:
                if pid not in pos_pids:
                    negatives.append(pid)
            qid2negatives[qid] = negatives

        # save qid2negatives to pickle file
        with open(os.path.join(work_dir, "qid2negatives.pkl"), "wb") as qid2negatives_fp:
            pickle.dump(qid2negatives, qid2negatives_fp, protocol=4)
    else:
        time.sleep(0)

    return None, None

    # enc_model.eval()
    # if query_model is None:
    #     query_enc_model = enc_model
    # else:
    #     query_model.eval()
    #     query_enc_model = query_model
    #
    # get_emb_lambda = get_representation_tensor if get_emb_lambda is None else get_emb_lambda
    #
    # vocab_id2token = vocab_id2token or dict((i, s) for s, i in tokenizer.get_vocab().items())
    #
    # # collection embeddings: get a dict_sparse for every document [token2int]
    #
    # # python_interpreter = "/home/shentao/anaconda3/envs/trans38/bin/"
    # # anserini_path = "/data/shentao/softwares/anserini"  # 0.14.0
    #
    # abs_output_dir = os.path.abspath(args.output_dir)
    # work_dir = os.path.join(abs_output_dir, "sparse_retrieval")
    #
    # saved_doc_emb_dir = os.path.join(work_dir, "sparse_emb_dir")
    # saved_doc_emb_path_template = os.path.join(saved_doc_emb_dir, "sparse_emb_{}.jsonl")
    # num_index_splits = 24
    #
    # if accelerator.is_local_main_process:
    #     if not dir_exists(work_dir):
    #         os.mkdir(work_dir)
    #     if not dir_exists(saved_doc_emb_dir):
    #         os.mkdir(saved_doc_emb_dir)
    # accelerator.wait_for_everyone()
    #
    # saved_index_dir = os.path.join(work_dir, "collection_index")
    #
    # if not dir_exists(saved_index_dir):
    #     with accelerator.main_process_first():
    #         passages_dataset = DatasetMacroPassages(
    #             "dev", args.data_dir, None, args, tokenizer, add_title=add_title)
    #     passages_dataloader = setup_eval_dataloader(args, passages_dataset, accelerator, use_accelerator=True)
    #
    #     if accelerator.is_local_main_process:
    #         collection_ptr = 0
    #         collection_index_fp_list = [
    #             open(saved_doc_emb_path_template.format(str(i)), "w") for i in range(num_index_splits)]
    #
    #     for batch_idx, batch in tqdm(
    #             enumerate(passages_dataloader), disable=not accelerator.is_local_main_process,
    #             total=len(passages_dataloader), desc=f"Getting passage vectors ..."):
    #         with torch.no_grad():
    #             embs = get_emb_lambda(enc_model(**batch)).contiguous()
    #         pids = batch["pids"]
    #
    #         pids, embs = accelerator.gather(pids), accelerator.gather(embs)
    #
    #         if accelerator.is_local_main_process:
    #             # pids_list.append(pids.cpu().numpy())
    #             # passage_vectors_list.append(embs.detach().cpu().numpy().astype("float32"))
    #             for pid, emb in zip(pids.cpu().numpy(), embs.detach().cpu().numpy()):
    #                 pid = int(pid)
    #                 dict_sparse = sparse_vector_to_dict(
    #                     emb, vocab_id2token, quantization_factor, dummy_token=tokenizer.unk_token)
    #                 # text = tokenizer.decode(batch["input_ids"].cpu().numpy().tolist())
    #                 index_dict = {"id": pid, "vector": dict_sparse,}  # "content": text,
    #                 collection_index_fp = collection_index_fp_list[collection_ptr%num_index_splits]
    #                 collection_index_fp.write(json.dumps(index_dict))
    #                 collection_index_fp.write(os.linesep)
    #                 collection_ptr+=1
    #     if accelerator.is_local_main_process:
    #         for collection_index_fp in collection_index_fp_list:
    #             collection_index_fp.close()
    #     accelerator.wait_for_everyone()
    #
    #     if accelerator.is_local_main_process:
    #         os.system(
    #             f"sh {anserini_path}/target/appassembler/bin/IndexCollection -collection JsonVectorCollection \
    #              -input {saved_doc_emb_dir} \
    #              -index {saved_index_dir} \
    #              -generator DefaultLuceneDocumentGenerator -impact -pretokenized \
    #              -threads {num_index_splits}")
    #         # -stemmer none
    #     else:
    #         time.sleep(0)
    #     accelerator.wait_for_everyone()
    #
    # # query embeddings
    # saved_query_emb_path = os.path.join(work_dir, "train_query_sparse_emb.tsv")
    # if not dir_exists(saved_query_emb_path):
    #     with accelerator.main_process_first():
    #         queries_dataset = DatasetFullRankQueries("train", args.data_dir, None, args, tokenizer, )
    #     queries_dataloader = setup_eval_dataloader(args, queries_dataset, accelerator, use_accelerator=True)
    #
    #     if accelerator.is_local_main_process:
    #         saved_query_emb_fp = open(saved_query_emb_path, "w")
    #     for batch_idx, batch in tqdm(
    #             enumerate(queries_dataloader), disable=not accelerator.is_local_main_process,
    #             total=len(queries_dataloader), desc=f"Getting query vectors ..."):
    #
    #         for k in list(batch.keys()):
    #             if k.endswith("_query"):
    #                 batch[k[:-6]] = batch.pop(k)
    #         with torch.no_grad():
    #             embs = get_emb_lambda(query_enc_model(**batch)).contiguous()
    #         qids = batch["qids"]
    #
    #         pids, embs = accelerator.gather(qids), accelerator.gather(embs)
    #         if accelerator.is_local_main_process:
    #             for qid, emb in zip(qids.cpu().numpy(), embs.detach().cpu().numpy()):
    #                 qid = int(qid)
    #                 dict_sparse = sparse_vector_to_dict(
    #                     emb, vocab_id2token, quantization_factor, dummy_token=tokenizer.unk_token)
    #                 saved_query_emb_fp.write(str(qid) + "\t" + dict_sparse_to_string(dict_sparse) + os.linesep)
    #     if accelerator.is_local_main_process:
    #         saved_query_emb_fp.close()
    # accelerator.wait_for_everyone()
    #
    # enc_model.train()
    # if query_model is not None:
    #     query_model.train()
    #
    # search_result_path = os.path.join(work_dir, "train_search_result.trec")
    # if not file_exists(search_result_path):
    #     if accelerator.is_local_main_process:
    #         os.system(
    #             f"sh {anserini_path}/target/appassembler/bin/SearchCollection -hits {hits} -parallelism 24 \
    #                  -index {saved_index_dir} \
    #                  -topicreader TsvInt -topics {saved_query_emb_path} \
    #                  -output {search_result_path} -format trec \
    #                  -impact -pretokenized"
    #         )
    #         # # Convert MS MARCO qrels to TREC qrels.
    #         # org_marco_train_qrels_path = os.path.join(
    #         #     args.data_dir, "msmarco/passage_ranking/qrels.train.tsv")
    #         # marco_train_qrels_path = os.path.join(work_dir, "qrels.train.trec")
    #         #
    #         # with open(marco_train_qrels_path, 'w') as fout:
    #         #     for line in open(org_marco_train_qrels_path):
    #         #         fout.write(line.replace('\t', ' '))
    #     else:
    #         time.sleep(0)
    # accelerator.wait_for_everyone()
    #
    # # load search_result_path back for remove positve pids
    # if accelerator.is_local_main_process:
    #     qrels = load_tsv(os.path.join(args.data_dir, "msmarco/passage_ranking/qrels.train.tsv"))
    #     qid2pos_pids = collections.defaultdict(set)
    #     for qrel in qrels:
    #         assert len(qrel) == 4
    #         qid, pid = int(qrel[0]), int(qrel[2])
    #         qid2pos_pids[qid].add(pid)
    #
    #     qid2all_pids = collections.defaultdict(list)
    #     with open(search_result_path) as search_result_fp:
    #         for line in search_result_fp:
    #             line_info = line.strip().split(" ")
    #             assert len(line_info) == 6
    #             qid, _, pid, rank, score, _method = line_info
    #             qid, pid, = int(qid), int(pid)
    #             qid2all_pids[qid].append(pid)
    #
    #     qid2negatives = dict()
    #     for qid in qid2all_pids:
    #         pos_pids = qid2pos_pids[qid]
    #         negatives = []
    #         for pid in qid2all_pids[qid]:
    #             if pid not in pos_pids:
    #                 negatives.append(pid)
    #         qid2negatives[qid] = negatives
    #
    #     # save qid2negatives to pickle file
    #     with open(os.path.join(work_dir, "qid2negatives.pkl"), "wb") as qid2negatives_fp:
    #         pickle.dump(qid2negatives, qid2negatives_fp, protocol=4)
    # else:
    #     qid2negatives = None
    # return qid2negatives
import json
import logging

import numpy as np
import time
import torch

from peach.datasets.marco.dataset_marco_eval import DatasetMacroPassages, DatasetFullRankQueries
from peach.base import *
from peach.common import file_exists, dir_exists, get_data_path_list
from tqdm import tqdm
import os
from peach.enc_utils.general import get_representation_tensor
from multiprocessing import Pool
from peach.enc_utils.eval_trec_file import evaluate_trec_file


def sparse_vector_to_dict(sparse_vec, vocab_id2token, quantization_factor, dummy_token):
    if isinstance(sparse_vec, tuple):
        idx, data = sparse_vec
    else:
        idx = np.nonzero(sparse_vec)[0]
        # then extract values:
        data = sparse_vec[idx]
    data = np.rint(data * quantization_factor).astype(int)

    dict_sparse = dict()

    for id_token, value_token in zip(idx, data):
        if value_token > 0:
            real_token = vocab_id2token[id_token]
            dict_sparse[real_token] = int(value_token)
    if len(dict_sparse.keys()) == 0:
        # print("empty input =>", id_)
        dict_sparse[dummy_token] = 1
        # in case of empty doc we fill with "[unused993]" token (just to fill
        # and avoid issues with anserini), in practice happens just a few times ...
    return dict_sparse


def process_one_vec_example(example):
    idx, pid, tuple_vec, vocab_id2token, quantization_factor, unk_token = example
    dict_sparse = sparse_vector_to_dict(
        tuple_vec, vocab_id2token, quantization_factor, dummy_token=unk_token)
    index_dict = {"id": pid, "vector": dict_sparse, }
    dump_str = json.dumps(index_dict) + os.linesep
    return idx, dump_str


def dict_sparse_to_string(dict_sparse, ):
    return " ".join(
        [" ".join([str(real_token)] * freq) for real_token, freq in dict_sparse.items()])


def evaluate_sparse_retreival(
    args, eval_dataset, enc_model, accelerator, global_step=None, tb_writer=None, save_prediction=False,
    key_metric_name="recall@100", delete_model=False, add_title=True, query_model=None,
    vocab_id2token=None, tokenizer=None, quantization_factor=100,
    anserini_path = "ANSERINI_HOME",  # 0.14.0
    get_emb_lambda=None, hits=1000,
    **kwargs,
):
    eval_dataset = eval_dataset or "dev"
    assert isinstance(eval_dataset, str)

    enc_model.eval()
    if query_model is None:
        query_enc_model = enc_model
    else:
        query_model.eval()
        query_enc_model = query_model

    get_emb_lambda = get_representation_tensor if get_emb_lambda is None else get_emb_lambda

    vocab_id2token = vocab_id2token or dict((i, s) for s, i in tokenizer.get_vocab().items())

    # collection embeddings: get a dict_sparse for every document [token2int]

    # python_interpreter = "/home/shentao/anaconda3/envs/trans38/bin/"
    # anserini_path = "/data/shentao/softwares/anserini"  # 0.14.0

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
                "train", args.data_dir, None, args, tokenizer, add_title=add_title)
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
                                          vocab_id2token, quantization_factor, tokenizer.unk_token,  # for compatible w/ Pool
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
    saved_query_emb_path = os.path.join(work_dir, f"{eval_dataset}_query_sparse_emb.tsv")
    if not file_exists(saved_query_emb_path):
        logging.info("Getting query vectors ...")
        with accelerator.main_process_first():
            queries_dataset = DatasetFullRankQueries(eval_dataset, args.data_dir, None, args, tokenizer, )
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
        search_result_filename = f"{eval_dataset}_search_result.trec" \
            if hits == 1000 else f"{eval_dataset}_search_result_hits{hits}.trec"
        search_result_path = os.path.join(work_dir, search_result_filename)
        if not file_exists(search_result_path):
            logging.info("Doing search ...")
            os.system(
                f"sh {anserini_path}/target/appassembler/bin/SearchCollection -hits {hits} -parallelism 80 \
                     -index {saved_index_dir} \
                     -topicreader TsvInt -topics {saved_query_emb_path} \
                     -output {search_result_path} -format trec \
                     -impact -pretokenized"
            )

        # Convert MS MARCO qrels to TREC qrels.
        org_marco_dev_qrels_path = os.path.join(
            args.data_dir, f"msmarco/passage_ranking/qrels.{eval_dataset}.tsv")
        marco_dev_qrels_path = os.path.join(work_dir, f"qrels.{eval_dataset}.trec")
        with open(marco_dev_qrels_path, 'w') as fout:
            for line in open(org_marco_dev_qrels_path):
                fout.write(line.replace('\t', ' '))

        # calculate score
        os.system(
            f"{anserini_path}/tools/eval/trec_eval.9.0.4/trec_eval -c -M 10 -m recip_rank \
                        {marco_dev_qrels_path} {search_result_path}")

        os.system(
            f"{anserini_path}/tools/eval/trec_eval.9.0.4/trec_eval -c -mrecall -mmap \
                        {marco_dev_qrels_path} {search_result_path}")

        # if eval_dataset == "test2019":
        os.system(
            f"{anserini_path}/tools/eval/trec_eval.9.0.4/trec_eval -c -mndcg_cut \
                                {marco_dev_qrels_path} {search_result_path}")

        eval_metrics = evaluate_trec_file(
            trec_file_path=search_result_path, qrels_file_path=org_marco_dev_qrels_path, )
        logger.info(f"step {global_step}: {eval_metrics}")
        key_metric = eval_metrics[key_metric_name]
        return key_metric, eval_metrics
    else:  # other processes
        return NEG_INF, {}

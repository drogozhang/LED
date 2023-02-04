import json
import numpy as np
import time
import torch

from peach.datasets.marco.dataset_marco_eval import DatasetMacroPassages, DatasetFullRankQueries
from peach.base import *
from peach.common import file_exists, dir_exists
from tqdm import tqdm
import os
import pickle
import faiss
from peach.enc_utils.general import get_representation_tensor
import collections
from peach.common import load_tsv


def get_hard_negative_by_dense_retrieval(
    args, eval_dataset, enc_model, accelerator, global_step=None, tb_writer=None, save_prediction=False,
    key_metric_name="recall@100", delete_model=False, add_title=True, query_model=None,
    tokenizer=None, faiss_mode="gpu",
    get_emb_lambda=None, hits=1000,
    **kwargs,
):
    # add_title = ("add_title" in kwargs) and kwargs["add_title"]
    enc_model.eval()
    if query_model is None:
        query_enc_model = enc_model
    else:
        query_model.eval()
        query_enc_model = query_model

    get_emb_lambda = get_representation_tensor if get_emb_lambda is None else get_emb_lambda

    abs_output_dir = os.path.abspath(args.output_dir)
    work_dir = os.path.join(abs_output_dir, "dense_retrieval")

    if accelerator.is_local_main_process:
        if not dir_exists(work_dir):
            os.mkdir(work_dir)
    accelerator.wait_for_everyone()

    collection_pids, collection_embs, all_qids, all_query_embs = None, None, None, None
    # passages
    dense_index_path = os.path.join(work_dir, "dense_index.pkl")  # collection_pids, collection_embs
    if not file_exists(dense_index_path):
        # get all vectors for collections
        with accelerator.main_process_first():
            passages_dataset = DatasetMacroPassages(
                "dev", args.data_dir, None, args, tokenizer, add_title=add_title)
        passages_dataloader = setup_eval_dataloader(
            args, passages_dataset, accelerator, use_accelerator=True)

        pids_list, passage_vectors_list = [], []
        for batch_idx, batch in tqdm(
                enumerate(passages_dataloader), disable=not accelerator.is_local_main_process,
                total=len(passages_dataloader), desc=f"Getting passage vectors ..."):
            pids = batch.pop("pids")
            with torch.no_grad():
                embs = get_emb_lambda(enc_model(**batch)).contiguous()

            pids, embs = accelerator.gather(pids), accelerator.gather(embs)

            if accelerator.is_local_main_process:
                pids_list.append(pids.cpu().numpy())
                passage_vectors_list.append(embs.detach().cpu().numpy().astype("float32"))
        accelerator.wait_for_everyone()
        if accelerator.is_local_main_process:
            collection_pids = np.concatenate(pids_list, axis=0)[:len(passages_dataset)]
            collection_embs = np.concatenate(passage_vectors_list, axis=0)[:len(passages_dataset)]
            with open(dense_index_path, "wb") as fp:
                pickle.dump([collection_pids, collection_embs], fp, protocol=4)
    accelerator.wait_for_everyone()

    # queries
    query_embs_path = os.path.join(work_dir, "train_query_embs.pkl")
    if not file_exists(query_embs_path):
        with accelerator.main_process_first():
            queries_dataset = DatasetFullRankQueries("train", args.data_dir, None, args, tokenizer, )
        queries_dataloader = setup_eval_dataloader(args, queries_dataset, accelerator, use_accelerator=True)
        qids_list, query_embs_list = [], []
        for batch_idx, batch in tqdm(
                enumerate(queries_dataloader), disable=not accelerator.is_local_main_process,
                total=len(queries_dataloader), desc=f"Getting query vectors ..."):
            qids = batch.pop("qids")
            for k in list(batch.keys()):
                if k.endswith("_query"):
                    batch[k[:-6]] = batch.pop(k)

            with torch.no_grad():
                embs = get_emb_lambda(query_enc_model(**batch)).contiguous()

            qids, embs = accelerator.gather(qids), accelerator.gather(embs)
            if accelerator.is_local_main_process:
                qids_list.append(qids.cpu().numpy())
                query_embs_list.append(embs.detach().cpu().numpy().astype("float32"))
        accelerator.wait_for_everyone()
        if accelerator.is_local_main_process:
            all_qids = np.concatenate(qids_list, axis=0)[:len(queries_dataset)]
            all_query_embs = np.concatenate(query_embs_list, axis=0)[:len(queries_dataset)]
            with open(query_embs_path, "wb") as fp:
                pickle.dump([all_qids, all_query_embs], fp, protocol=4)
    accelerator.wait_for_everyone()

    if delete_model:  # delete model to free space for faiss
        free_model(enc_model, accelerator)
        free_model(query_model, accelerator)
    else:
        enc_model.train()
        if query_model is not None:
            query_model.train()
    free_memory()

    # do dense retrieval
    if accelerator.is_local_main_process:
        if collection_pids is None or collection_embs is None:
            with open(dense_index_path, "rb") as fp:
                collection_pids, collection_embs = pickle.load(fp)
        if all_qids is None or all_query_embs is None:
            with open(query_embs_path, "rb") as fp:
                all_qids, all_query_embs = pickle.load(fp)

        # faiss stuff to build index
        # faiss_mode = "gpu"
        logger.info(f"Using faiss_mode {faiss_mode} ...")
        t0 = time.time()
        logger.info("Building faiss index ...")
        dim = collection_embs.shape[-1]
        if faiss_mode == "cpu":
            index_engine = faiss.IndexFlatIP(dim)
            index_engine.add(collection_embs)
        elif faiss_mode == "gpu":
            logger.info(f"found {faiss.get_num_gpus()} gpus in faiss ...")
            res = faiss.StandardGpuResources()
            index_flat = faiss.IndexFlatIP(dim)
            index_engine = faiss.index_cpu_to_gpu(res, 0, index_flat)
            index_engine.add(collection_embs)
        else:
            raise NotImplementedError(faiss_mode)
        logger.info(f"Using {time.time() - t0}sec to build index for {collection_embs.shape[0]} docs")

        def _batch_search(_index, _queries, _batch_size, _k):
            Ds, Is = [], []
            for _start in tqdm(range(0, _queries.shape[0], _batch_size)):
                D, I = _index.search(_queries[_start: _start + _batch_size], k=_k)
                Ds.append(D)
                Is.append(I)
            return np.concatenate(Ds, axis=0), np.concatenate(Is, axis=0)

        t0 = time.time()
        logger.info("Doing faiss search ...")
        assert index_engine.is_trained
        D, I = _batch_search(index_engine, all_query_embs, _batch_size=64, _k=hits)
        del index_engine
        logger.info(f"Using {time.time() - t0}sec to complete search for {all_query_embs.shape[0]} queries")

        # save to qid2negatives.pkl
        qrels = load_tsv(os.path.join(args.data_dir, "msmarco/passage_ranking/qrels.train.tsv"))
        qid2pos_pids = collections.defaultdict(set)
        for qrel in qrels:
            assert len(qrel) == 4
            qid, pid = int(qrel[0]), int(qrel[2])
            qid2pos_pids[qid].add(pid)

        # metric_ranking = datasets.load_metric("peach/metrics/ranking.py")
        # metric_ranking.qids_list = []
        qid2negatives = dict()
        for q_index, qid in tqdm(enumerate(all_qids), desc="Calculating metrics ..."):
            qid = int(qid)
            gold_pids = qid2pos_pids[qid]
            top_pids = [int(collection_pids[int(p_index)]) for p_index in I[q_index]]
            # top_references = np.array([int(pid in gold_pids) for pid in top_pids], dtype="int64")
            # top_scores = D[q_index]
            # metric_ranking.add_batch(predictions=top_scores, references=top_references)
            # metric_ranking.qids_list.extend([qid] * 1000)
            negative_pids = [pid for pid in top_pids if pid not in gold_pids]
            qid2negatives[qid] = negative_pids

        # save qid2negatives to pickle file
        with open(os.path.join(work_dir, "qid2negatives.pkl"), "wb") as qid2negatives_fp:
            pickle.dump(qid2negatives, qid2negatives_fp, protocol=4)
    else:
        qid2negatives = None
    return qid2negatives
















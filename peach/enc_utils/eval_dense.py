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


def evaluate_dense_retreival(
    args, eval_dataset, enc_model, accelerator, global_step=None, tb_writer=None, save_prediction=False,
    key_metric_name="recall@100", delete_model=False, add_title=True, query_model=None,
    tokenizer=None, faiss_mode="gpu",
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
                "train", args.data_dir, None, args, tokenizer, add_title=add_title)
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
    query_embs_path = os.path.join(work_dir, f"{eval_dataset}_query_embs.pkl")
    if not file_exists(query_embs_path):
        with accelerator.main_process_first():
            queries_dataset = DatasetFullRankQueries(eval_dataset, args.data_dir, None, args, tokenizer, )
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
        metric_ranking = datasets.load_metric("peach/metrics/ranking_v2.py")
        metric_ranking.qids_list = []
        search_results = []
        for q_index, qid in tqdm(enumerate(all_qids), desc="Calculating metrics ..."):
            qid = int(qid)
            # gold_pids = qid2pos_pids[qid]
            top_pids = [int(collection_pids[int(p_index)]) for p_index in I[q_index]]
            top_scores = D[q_index]
            metric_ranking.add_batch(predictions=top_scores, references=top_pids)
            metric_ranking.qids_list.extend([qid] * len(top_pids))
            search_results.extend(
                [(qid, pid, idx + 1, score)
                 for idx, (pid, score) in enumerate(zip(top_pids, top_scores.tolist()))])
        eval_metrics = metric_ranking.compute(
            qrels_path=os.path.join(args.data_dir, f"msmarco/passage_ranking/qrels.{eval_dataset}.tsv"),
            group_labels=metric_ranking.qids_list, )
        """
        qrels = load_tsv(os.path.join(args.data_dir, "msmarco/passage_ranking/qrels.dev.tsv"))
        qid2pos_pids = collections.defaultdict(set)
        for qrel in qrels:
            assert len(qrel) == 4
            qid, pid = int(qrel[0]), int(qrel[2])
            qid2pos_pids[qid].add(pid)

        metric_ranking = datasets.load_metric("peach/metrics/ranking.py")
        metric_ranking.qids_list = []
        # qid2negatives = dict()
        search_results = []
        for q_index, qid in tqdm(enumerate(all_qids), desc="Calculating metrics ..."):
            qid = int(qid)
            gold_pids = qid2pos_pids[qid]
            top_pids = [int(collection_pids[int(p_index)]) for p_index in I[q_index]]
            top_references = np.array([int(pid in gold_pids) for pid in top_pids], dtype="int64")
            top_scores = D[q_index]
            metric_ranking.add_batch(predictions=top_scores, references=top_references)
            metric_ranking.qids_list.extend([qid] * len(top_pids))
            # negative_pids = [pid for pid in top_pids if pid not in gold_pids]
            # qid2negatives[qid] = negative_pids
            search_results.extend(
                [(qid, pid, idx+1, score)
                 for idx, (pid, score) in enumerate(zip(top_pids, top_scores.tolist()))])

        eval_metrics = metric_ranking.compute(group_labels=metric_ranking.qids_list)
        """

        logger.info(f"step {global_step}: {eval_metrics}")
        if global_step is not None:
            for key, val in eval_metrics.items():
                if isinstance(val, (float, int)):
                    tb_writer.add_scalar(f"eval_in_train-{key}", val, global_step)
        key_metric = eval_metrics[key_metric_name]

        search_result_filename = f"{eval_dataset}_search_result.trec" \
            if hits == 1000 else f"{eval_dataset}_search_result_hits{hits}.trec"
        search_result_path = os.path.join(work_dir, search_result_filename)
        with open(search_result_path, "w") as fp:
            for qid, pid, rank, score in search_results:
                fp.write(f"{qid} Q0 {pid} {rank} {score} DenseRetrieval")
                fp.write(os.linesep)
        # # save qid2negatives to pickle file
        # with open(os.path.join(work_dir, "qid2negatives.pkl"), "wb") as qid2negatives_fp:
        #     pickle.dump(qid2negatives, qid2negatives_fp, protocol=4)
    else:
        key_metric = NEG_INF
        eval_metrics = {}
    accelerator.wait_for_everyone()
    return key_metric, eval_metrics


def evaluate_dense_retreival_lagacy(
    args, eval_dataset, enc_model, accelerator, global_step=None, tb_writer=None, save_prediction=False,
    key_metric_name="recall@100", delete_model=False, add_title=True, query_model=None,
    tokenizer=None, faiss_mode="gpu",
    **kwargs,
):
    enc_model.eval()
    if query_model is None:
        query_enc_model = enc_model
    else:
        query_model.eval()
        query_enc_model = query_model

    saved_emb_path = os.path.join(args.output_dir, "embeddings.pkl")
    file_cache_flag = False

    if not file_exists(saved_emb_path):

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
                embs = get_representation_tensor(enc_model(**batch)).contiguous()

            pids, embs = accelerator.gather(pids), accelerator.gather(embs)

            if accelerator.is_local_main_process:
                pids_list.append(pids.cpu().numpy())
                passage_vectors_list.append(embs.detach().cpu().numpy().astype("float32"))
        accelerator.wait_for_everyone()

        # get all vector for query and query the index
        with accelerator.main_process_first():
            queries_dataset = DatasetFullRankQueries("dev", args.data_dir, None, args, tokenizer, )
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
                embs = get_representation_tensor(query_enc_model(**batch)).contiguous()

            qids, embs = accelerator.gather(qids), accelerator.gather(embs)
            if accelerator.is_local_main_process:
                qids_list.append(qids.cpu().numpy())
                query_embs_list.append(embs.detach().cpu().numpy().astype("float32"))
        accelerator.wait_for_everyone()

    if delete_model:  # delete model to free space for faiss
        free_model(enc_model, accelerator)
        free_model(query_model, accelerator)
    else:
        enc_model.train()
        if query_model is not None:
            query_model.train()
    free_memory()

    if accelerator.is_local_main_process:
        if file_exists(saved_emb_path):
            logger.info("Reading saved embeddings ...")
            with open(saved_emb_path, "rb") as fp:
                collection_pids, collection_embs, all_qids, all_query_embs = pickle.load(fp)
            queries_dataset = DatasetFullRankQueries("dev", args.data_dir, None, args, tokenizer, )
        else:
            collection_pids = np.concatenate(pids_list, axis=0)[:len(passages_dataset)]
            collection_embs = np.concatenate(passage_vectors_list, axis=0)[:len(passages_dataset)]

            all_qids = np.concatenate(qids_list, axis=0)[:len(queries_dataset)]
            all_query_embs = np.concatenate(query_embs_list, axis=0)[:len(queries_dataset)]

            if file_cache_flag:
                with open(saved_emb_path, "wb") as fp:
                    pickle.dump([collection_pids, collection_embs, all_qids, all_query_embs], fp, protocol=4)

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

        def _batch_search(_index, _queries, _batch_size, _k=1000):
            Ds, Is = [], []
            for _start in tqdm(range(0, _queries.shape[0], _batch_size)):
                D, I = _index.search(_queries[_start: _start + _batch_size], k=_k)
                Ds.append(D)
                Is.append(I)
            return np.concatenate(Ds, axis=0), np.concatenate(Is, axis=0)

        t0 = time.time()
        logger.info("Doing faiss search ...")
        assert index_engine.is_trained
        D, I = _batch_search(index_engine, all_query_embs, _batch_size=64, _k=1000)
        del index_engine
        logger.info(f"Using {time.time() - t0}sec to complete search for {all_query_embs.shape[0]} queries")

        # calculate metrics
        metric_ranking = datasets.load_metric("peach/metrics/ranking.py")
        metric_ranking.qids_list = []
        for q_index, qid in tqdm(enumerate(all_qids), desc="Calculating metrics ..."):
            qid = int(qid)
            gold_pids = queries_dataset.qid2pids[qid]
            top_references = np.array([int(int(collection_pids[int(p_index)]) in gold_pids) for p_index in I[q_index]],
                                      dtype="int64")
            top_scores = D[q_index]
            metric_ranking.add_batch(
                predictions=top_scores, references=top_references
            )
            metric_ranking.qids_list.extend([qid] * 1000)

        eval_metrics = metric_ranking.compute(group_labels=metric_ranking.qids_list)

        logger.info(f"step {global_step}: {eval_metrics}")
        if global_step is not None:
            for key, val in eval_metrics.items():
                if isinstance(val, (float, int)):
                    tb_writer.add_scalar(f"eval_in_train-{key}", val, global_step)
        key_metric = eval_metrics[key_metric_name]
    else:
        key_metric = NEG_INF
        eval_metrics = {}

    accelerator.wait_for_everyone()

    return key_metric, eval_metrics

def evaluate_dense_retrieval_for_adore(
    args, eval_dataset, enc_model, accelerator, global_step=None, tb_writer=None, save_prediction=False,
    key_metric_name="recall@100", delete_model=False, add_title=True, query_model=None,
    tokenizer=None, faiss_mode="gpu", get_emb_lambda=None, hits=1000,
    **kwargs, ):

    enc_model.eval()
    if query_model is None:
        query_enc_model = enc_model
    else:
        query_model.eval()
        query_enc_model = query_model
    stage2_saved_emb_path = os.path.join(args.stage2_output_dir, "embeddings.pkl")
    
    if file_exists(stage2_saved_emb_path):
        logger.info("Reading saved embeddings ...")
        with open(stage2_saved_emb_path, "rb") as fp:
            # TODO modify for future.
            try:
                collection_pids, collection_embs, _all_qids, _all_query_embs = pickle.load(fp)
            except:
                collection_pids, collection_embs = pickle.load(fp)  # latest Tao's code 

    get_emb_lambda = get_representation_tensor if get_emb_lambda is None else get_emb_lambda

    abs_output_dir = os.path.abspath(args.output_dir)
    work_dir = os.path.join(abs_output_dir, "dense_retrieval")


    saved_search_result_path = os.path.join(args.output_dir, "search_result.pkl")
    file_cache_flag = False
    
    # get all vector for query and query the index
    with accelerator.main_process_first():
        queries_dataset = DatasetFullRankQueries("dev", args.data_dir, None, args, tokenizer, )
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
            embs = get_representation_tensor(query_enc_model(**batch)).contiguous()
        
        qids, embs = accelerator.gather(qids), accelerator.gather(embs)
        if accelerator.is_local_main_process:
            qids_list.append(qids.cpu().numpy())
            query_embs_list.append(embs.detach().cpu().numpy().astype("float32"))
    accelerator.wait_for_everyone()

    if delete_model:  # delete model to free space for faiss
        free_model(enc_model, accelerator)
        free_model(query_model, accelerator)
    else:
        enc_model.train()
        if query_model is not None:
            query_model.train()
    free_memory()

    if accelerator.is_local_main_process:
        all_qids = np.concatenate(qids_list, axis=0)[:len(queries_dataset)]
        all_query_embs = np.concatenate(query_embs_list, axis=0)[:len(queries_dataset)]
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

        def _batch_search(_index, _queries, _batch_size, _k=hits):
            Ds, Is = [], []
            for _start in tqdm(range(0, _queries.shape[0], _batch_size)):
                D, I = _index.search(_queries[_start: _start+_batch_size], k=_k)
                Ds.append(D) # search scores [batch_size, _k]
                Is.append(I) # pids [batch_size, _k]
            return np.concatenate(Ds, axis=0), np.concatenate(Is, axis=0)

        t0 = time.time()
        logger.info("Doing faiss search ...")
        assert index_engine.is_trained
        D, I = _batch_search(index_engine, all_query_embs, _batch_size=64, _k=hits)  # [query_num, top_k] for score, for collection_id
        del index_engine
        logger.info(f"Using {time.time() - t0}sec to complete search for {all_query_embs.shape[0]} queries")
        # KZ 4/2/2022 save dense search results
        if file_cache_flag:
            with open(saved_search_result_path, "wb") as fp:
                pickle.dump([D, I, all_qids], fp, protocol=4) # dense_1000_scores, dense_1000_pids, dense_qids
                logger.info("FAISS search results saved.")

        # calculate metrics
        metric_ranking = datasets.load_metric("peach/metrics/ranking.py") 
        metric_ranking.qids_list = [] 
        search_results = []
        for q_index, qid in tqdm(enumerate(all_qids), desc="Calculating metrics ..."):
            qid = int(qid)
            gold_pids = queries_dataset.qid2pids[qid]
            top_pids = [int(collection_pids[int(p_index)]) for p_index in I[q_index]]
            top_references = np.array([int(int(collection_pids[int(p_index)]) in gold_pids) for p_index in I[q_index]],
                                      dtype="int64")
            top_scores = D[q_index]
            metric_ranking.add_batch(
                predictions=top_scores, references=top_references
            )
            metric_ranking.qids_list.extend([qid] * 1000)
            search_results.extend(
                [(qid, pid, idx+1, score)
                 for idx, (pid, score) in enumerate(zip(top_pids, top_scores.tolist()))])

        eval_metrics = metric_ranking.compute(group_labels=metric_ranking.qids_list)

        logger.info(f"step {global_step}: {eval_metrics}")
        if global_step is not None:
            for key, val in eval_metrics.items():
                if isinstance(val, (float, int)):
                    tb_writer.add_scalar(f"eval_in_train-{key}", val, global_step)
        key_metric = eval_metrics[key_metric_name]

        search_result_path = os.path.join(work_dir, "search_result.trec")
        with open(search_result_path, "w") as fp:
            for qid, pid, rank, score in search_results:
                fp.write(f"{qid} Q0 {pid} {rank} {score} ADORE_DenseRetrieval")
                fp.write(os.linesep)
    else:
        key_metric = NEG_INF
        eval_metrics = {}

    accelerator.wait_for_everyone()

    return key_metric, eval_metrics

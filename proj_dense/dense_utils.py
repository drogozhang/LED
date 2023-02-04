import json
import tqdm
import accelerate

from peach.base import *
from peach.common import save_jsonl, save_pickle, file_exists
import pickle
from scipy.special import softmax


from peach.datasets.marco.dataset_marco_eval import DatasetMacroPassages, DatasetFullRankQueries
from peach.enc_utils.general import get_representation_tensor

import faiss
import time


def get_hard_negative_by_retriever(
    args, eval_dataset, enc_model, accelerator, global_step=None, tb_writer=None, save_prediction=False,
    delete_model=False, add_title=True, query_model=None,
    tokenizer=None, faiss_mode="gpu", **kwargs,
):
    # add_title = ("add_title" in kwargs) and kwargs["add_title"]
    enc_model.eval()
    if query_model is None:
        query_enc_model = enc_model
    else:
        query_model.eval()
        query_enc_model = query_model

    saved_emb_path = os.path.join(args.output_dir, "embeddings_hard_neg.pkl")
    qid2negatives_path = os.path.join(args.output_dir, "qid2negatives.pkl")
    file_cache_flag = True

    if file_exists(qid2negatives_path):
        logger.info("Reading hard negative embeddings ...")
        with open(qid2negatives_path, "rb") as fp:
            qid2negatives = pickle.load(fp)
            return qid2negatives


    if not file_exists(saved_emb_path):

        # get all vectors for collections
        with accelerator.main_process_first():
            passages_dataset = DatasetMacroPassages("train", args.data_dir, None, args, tokenizer, add_title=add_title)
        passages_dataloader = setup_eval_dataloader(args, passages_dataset, accelerator, use_accelerator=True)

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

        # TRAIN: get all vector for query and query the index
        with accelerator.main_process_first():
            train_queries_dataset = DatasetFullRankQueries("train", args.data_dir, None, args, tokenizer, )
        train_queries_dataloader = setup_eval_dataloader(args, train_queries_dataset, accelerator, use_accelerator=True)

        train_qids_list, train_query_embs_list = [], []
        for batch_idx, batch in tqdm(
                enumerate(train_queries_dataloader), disable=not accelerator.is_local_main_process,
                total=len(train_queries_dataloader), desc=f"Getting query vectors ..."):
            
            qids = batch.pop("qids")
            for k in list(batch.keys()):
                if k.endswith("_query"):
                    batch[k[:-6]] = batch.pop(k)
            
            with torch.no_grad():
                embs = get_representation_tensor(query_enc_model(**batch)).contiguous()
            
            qids, embs = accelerator.gather(qids), accelerator.gather(embs)
            if accelerator.is_local_main_process:
                train_qids_list.append(qids.cpu().numpy())
                train_query_embs_list.append(embs.detach().cpu().numpy().astype("float32"))

        # DEV: get all vector for query and query the index # why dev? query is fast, doesn't matter
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
                collection_pids, collection_embs, all_train_qids, all_train_query_embs, all_qids, all_query_embs = pickle.load(fp)
            train_queries_dataset = DatasetFullRankQueries("train", args.data_dir, None, args, tokenizer, )
            queries_dataset = DatasetFullRankQueries("dev", args.data_dir, None, args, tokenizer, )
        else:
            collection_pids = np.concatenate(pids_list, axis=0)[:len(passages_dataset)]
            collection_embs = np.concatenate(passage_vectors_list, axis=0)[:len(passages_dataset)]

            all_train_qids = np.concatenate(train_qids_list, axis=0)[:len(train_queries_dataset)]
            all_train_query_embs = np.concatenate(train_query_embs_list, axis=0)[:len(train_queries_dataset)]

            all_qids = np.concatenate(qids_list, axis=0)[:len(queries_dataset)]
            all_query_embs = np.concatenate(query_embs_list, axis=0)[:len(queries_dataset)]

            if file_cache_flag:
                logger.info("Saving saved embeddings ...")
                with open(saved_emb_path, "wb") as fp:
                    pickle.dump([collection_pids, collection_embs, all_train_qids, all_train_query_embs, all_qids, all_query_embs], fp, protocol=4)
        # all_qids, all_query_embs = all_qids[:10000], all_query_embs[:10000]
        free_memory()

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

        logger.info("Doing faiss search ...")
        def _batch_search(_index, _queries, _batch_size, _k=1000):
            Ds, Is = [], []
            for _start in tqdm(range(0, _queries.shape[0], _batch_size)):
                D, I = _index.search(_queries[_start: _start+_batch_size], k=_k)
                Ds.append(D)
                Is.append(I)
            return np.concatenate(Ds, axis=0), np.concatenate(Is, axis=0)
        train_D, train_I = _batch_search(index_engine, all_train_query_embs, _batch_size=64, _k=1000)  # kwargs["num_hard_negatives"]+16
        D, I = _batch_search(index_engine, all_query_embs, _batch_size=64, _k=1000)
        del index_engine

        # get metrics
        metric_ranking = datasets.load_metric("peach/metrics/ranking.py")
        metric_ranking.qids_list = []
        for q_index, qid in tqdm(enumerate(all_qids), desc="Calculating metrics ..."):
            qid = int(qid)
            gold_pids = queries_dataset.qid2pids[qid]
            top_references = np.array(
                [int(int(collection_pids[int(p_index)]) in gold_pids) for p_index in I[q_index]], dtype="int64")
            top_scores = D[q_index]
            metric_ranking.add_batch(
                predictions=top_scores, references=top_references
            )
            metric_ranking.qids_list.extend([qid] * 1000)

        eval_metrics = metric_ranking.compute(group_labels=metric_ranking.qids_list)
        logger.info(f"The metrics at the end of a stage: {eval_metrics}")

        # sample negatives
        metric_ranking = datasets.load_metric("peach/metrics/ranking.py")
        metric_ranking.qids_list = []

        qid2negatives = {}
        for q_index, qid in tqdm(enumerate(all_train_qids), desc="Getting negatives ..."):
            qid = int(qid)
            gold_pids = train_queries_dataset.qid2pids[qid]

            top_references = np.array([int(int(collection_pids[int(p_index)]) in gold_pids) for p_index in train_I[q_index]], dtype="int64")
            top_scores = train_D[q_index]
            if q_index < 5000:
                metric_ranking.add_batch(
                    predictions=top_scores, references=top_references
                )
                metric_ranking.qids_list.extend([qid] * 1000)

            negative_pids = []
            for p_index in train_I[q_index]:
                pid = int(collection_pids[int(p_index)])
                if pid not in gold_pids:
                    negative_pids.append(pid)
                if len(negative_pids) == kwargs["num_hard_negatives"]:
                    break
            qid2negatives[qid] = negative_pids

        eval_metrics = metric_ranking.compute(group_labels=metric_ranking.qids_list)
        logger.info(f"!!!: The train metrics at the end of a stage: {eval_metrics}")
    else:
        time.sleep(1200)  # sleep 1200s
        qid2negatives = None

    if not file_exists(qid2negatives_path) and file_cache_flag:
        logger.info("Saving hard negative embeddings ...")
        with open(qid2negatives_path, "wb") as fp:
            pickle.dump(qid2negatives, fp, protocol=4)

    accelerator.wait_for_everyone()
    enc_model.train()
    return qid2negatives

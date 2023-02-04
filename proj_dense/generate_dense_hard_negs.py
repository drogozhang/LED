from email.policy import default
import json
from numpy import require
import tqdm
import accelerate
from transformers import AutoModel

from peach.base import *
from peach.common import save_jsonl, save_pickle, file_exists
import pickle
from scipy.special import softmax

from peach.datasets.marco.dataset_marco_passages import DatasetMarcoPassagesRanking

from peach.datasets.marco.dataset_marco_eval import DatasetMacroPassages, DatasetFullRankQueries
from peach.enc_utils.general import get_representation_tensor

import faiss
import time


def prepare_logger(args, accelerator):
    if accelerator.is_local_main_process:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S")
        fh = logging.FileHandler(os.path.join(args.output_dir, "generate-hard-neg-logging.txt"))  # use FileHandler to file
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        ch = logging.StreamHandler()  # use StreamHandler to console
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)  # add to Handler
        logger.addHandler(fh)
    else:
        logger.setLevel(logging.WARNING)
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S")
        fh = logging.FileHandler(os.path.join(args.output_dir, f"hard-neg-logging_subprocess-{dist.get_rank()}.txt"))  # use FileHandler to file
        fh.setLevel(logging.WARNING)
        fh.setFormatter(formatter)
        ch = logging.StreamHandler()  # use StreamHandler to console
        ch.setLevel(logging.WARNING)
        ch.setFormatter(formatter)
        logger.addHandler(ch)  # add to Handler
        logger.addHandler(fh)


    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    # logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    # if accelerator.is_local_main_process:
    #     datasets.utils.logging.set_verbosity_warning()
    #     transformers.utils.logging.set_verbosity_info()
    # else:
    #     datasets.utils.logging.set_verbosity_error()
    #     transformers.utils.logging.set_verbosity_error()
    logger.info(accelerator.state)


def add_model_configs(parser):
    # copy from huggingface
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument("--train_batch_size", default=32, type=int, help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=12, type=int, help="Total batch size for eval.")
    pass


def get_hard_negative_by_retriever(
    args, enc_model, accelerator, delete_model=False, add_title=True, query_model=None,
    tokenizer=None, faiss_mode="gpu", **kwargs,):
    enc_model.eval()
    if query_model is None:
        query_enc_model = enc_model
    else:
        query_model.eval()
        query_enc_model = query_model

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

    if delete_model:  # delete model to free space for faiss
        free_model(enc_model, accelerator)
        free_model(query_model, accelerator)
    else:
        enc_model.train()
        if query_model is not None:
            query_model.train()
    free_memory()

    if accelerator.is_local_main_process:
        
        collection_pids = np.concatenate(pids_list, axis=0)[:len(passages_dataset)]
        collection_embs = np.concatenate(passage_vectors_list, axis=0)[:len(passages_dataset)]

        all_train_qids = np.concatenate(train_qids_list, axis=0)[:len(train_queries_dataset)]
        all_train_query_embs = np.concatenate(train_query_embs_list, axis=0)[:len(train_queries_dataset)]
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
        del index_engine

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

    accelerator.wait_for_everyone()
    return qid2negatives


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_load_type", type=str, default="disk", choices=["disk", "memory"])
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--num_negatives", type=int, default=200)
    parser.add_argument("--num_proc", type=int, default=4)
    parser.add_argument("--overwrite_save_path", action='store_true')

    add_model_configs(parser)

    args = parser.parse_args()

    qid2negatives_path = os.path.join(args.output_dir, f"top_{args.num_negatives}_qid2negatives.pkl")

    if file_exists(qid2negatives_path):
        if not args.overwrite_save_path:
            raise ValueError("Output static hard negative path ({}) already exists. Use --overwrite_save_path to overcome.".format(args.overwrite_save_path))

    accelerator = Accelerator(
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=True)])

    prepare_logger(args, accelerator)

    args.per_device_eval_batch_size = args.eval_batch_size // accelerator.num_processes
    config, tokenizer = load_config_and_tokenizer(args)

    encoder = AutoModel.from_pretrained(args.model_name_or_path, config=config)
    encoder = accelerator.prepare(encoder)

    qid2negatives = get_hard_negative_by_retriever(args, encoder, accelerator,
        tokenizer=tokenizer, num_hard_negatives=args.num_negatives, add_title=True,
        delete_model=True,  # delete model in case of GPU OOM during faiss (need 32G)
        faiss_mode="gpu",)

    if qid2negatives is not None:
        logger.info("Saving qid2hard_negative to {}".format(qid2negatives_path))
        with open(qid2negatives_path, "wb") as fp:
            pickle.dump(qid2negatives, fp, protocol=4)


if __name__ == '__main__':
    main()

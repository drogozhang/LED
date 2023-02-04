from email.policy import default
import sys
import csv
import json
import tqdm
import time
import pickle
import collections

import accelerate
from numpy import require
from proj_dense.generate_dense_hard_negs import prepare_logger
from transformers import AutoModel
from peach.base import *
from peach.common import save_jsonl, save_pickle, file_exists, dir_exists
from peach.enc_utils.general import get_representation_tensor
from peach.enc_utils.eval_sparse import sparse_vector_to_dict, dict_sparse_to_string
from proj_sparse.modeling_splade_series import add_model_hyperparameters, DistilBertSpladeEnocder, \
    BertSpladeEnocder, ConSpladeEnocder, RobertaSpladeEnocder
from peach.datasets.marco.dataset_marco_passages import DatasetMarcoPassagesRanking
from peach.datasets.marco.dataset_marco_eval import (
    DatasetMacroPassages,
    DatasetFullRankQueries,
)


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

# may encoder error (special display like:
# ::::::::::::::
# FILE_PATH
# ::::::::::::::
def load_trec(trec_file_path):
    logger.info("Loading trec search results...")
    qid2pids = dict()  # {qid: [pid1, pid2]}
    cnt = 0
    with open(trec_file_path) as fp:
        line = fp.readline()
        while line:
            line_info = line.strip().split(" ")
            if len(line_info) == 6:
                qid, _, pid, rank, score, _method = line_info
                qid, pid, = int(qid), int(pid)
                if qid not in qid2pids.keys():  # initialize
                    qid2pids[qid] = []
                qid2pids[qid].append(pid)
                cnt += 1
            if cnt % 1e6 == 0:
                logger.info(f"Processed lines: {cnt}")

            line = fp.readline()

    return qid2pids

def get_hard_negative_by_retriever(
    args, enc_model, accelerator, add_title=True, query_model=None,
    vocab_id2token=None, tokenizer=None, quantization_factor=100,
    **kwargs,
):
    enc_model.eval()
    if query_model is None:
        query_enc_model = enc_model
    else:
        query_model.eval()
        query_enc_model = query_model

    vocab_id2token = vocab_id2token or tokenizer.get_vocab()

    # python_interpreter = "/home/shentao/anaconda3/envs/trans38/bin/"
    anserini_path = "/relevance2-nfs/v-zhangkai/Workspace/anserini"  # 0.14.0

    abs_output_dir = os.path.abspath(args.output_dir)
    work_dir = os.path.join(abs_output_dir, "sparse_retrieval")  # save with do_prediction index

    saved_doc_emb_dir = os.path.join(work_dir, "train_sparse_emb_dir")
    saved_doc_emb_path_template = os.path.join(saved_doc_emb_dir, "train_sparse_emb_{}.jsonl")
    num_index_splits = 24

    if accelerator.is_local_main_process:
        if not dir_exists(work_dir):
            os.mkdir(work_dir)
        if not dir_exists(saved_doc_emb_dir):
            os.mkdir(saved_doc_emb_dir)
    accelerator.wait_for_everyone()

    saved_index_dir = os.path.join(work_dir, "collection_index")

    # load train data
    if not dir_exists(saved_index_dir):
        with accelerator.main_process_first():
            passages_dataset = DatasetMacroPassages("train", args.data_dir, None, args, tokenizer, add_title=add_title)
        passages_dataloader = setup_eval_dataloader(args, passages_dataset, accelerator, use_accelerator=True)

        if accelerator.is_local_main_process:
            collection_ptr = 0
            collection_index_fp_list = [
                open(saved_doc_emb_path_template.format(str(i)), "w") for i in range(num_index_splits)]

        for batch_idx, batch in tqdm(
                enumerate(passages_dataloader), disable=not accelerator.is_local_main_process,
                total=len(passages_dataloader), desc=f"Getting passage vectors ..."):
            with torch.no_grad():
                embs = get_representation_tensor(enc_model(**batch)).contiguous()

            pids = batch["pids"]
            pids, embs = accelerator.gather(pids), accelerator.gather(embs)

            if accelerator.is_local_main_process:
                # pids_list.append(pids.cpu().numpy())
                # passage_vectors_list.append(embs.detach().cpu().numpy().astype("float32"))
                for pid, emb in zip(pids.cpu().numpy(), embs.detach().cpu().numpy()):
                    pid = int(pid)
                    dict_sparse = sparse_vector_to_dict(
                        emb, vocab_id2token, quantization_factor, dummy_token=tokenizer.unk_token)
                    # text = tokenizer.decode(batch["input_ids"].cpu().numpy().tolist())
                    index_dict = {"id": pid, "vector": dict_sparse,}  # "content": text,
                    collection_index_fp = collection_index_fp_list[collection_ptr%num_index_splits]
                    collection_index_fp.write(json.dumps(index_dict))
                    collection_index_fp.write(os.linesep)
                    collection_ptr += 1
        if accelerator.is_local_main_process:
            for collection_index_fp in collection_index_fp_list:
                collection_index_fp.close()
        accelerator.wait_for_everyone()

        if accelerator.is_local_main_process:
            os.system(
                f"sh {anserini_path}/target/appassembler/bin/IndexCollection -collection JsonVectorCollection \
                 -input {saved_doc_emb_dir} \
                 -index {saved_index_dir} \
                 -generator DefaultLuceneDocumentGenerator -impact -pretokenized \
                 -threads {num_index_splits}")
            # -stemmer none
        else:
            time.sleep(0)
        accelerator.wait_for_everyone()

    # train_query embeddings
    file_cnt = 1
    line_cnt = 0
    per_file_query_num_limit = 5e4  # 50k ~11 files.
    expect_query_file_num = 11 # SET BY MYSELF
    expect_query_end_file_name = os.path.join(work_dir, f"train_query_sparse_emb_{expect_query_file_num}.tsv")

    with accelerator.main_process_first():
        train_queries_dataset = DatasetFullRankQueries("train", args.data_dir, None, args, tokenizer, )
    if not file_exists(expect_query_end_file_name):
    # if not file_exists(expect_query_end_file_name):
        query_sparse_file_name = f"train_query_sparse_emb_{file_cnt}.tsv"
        saved_query_emb_path = os.path.join(work_dir, query_sparse_file_name)
        train_queries_dataloader = setup_eval_dataloader(args, train_queries_dataset, accelerator, use_accelerator=True)

        if accelerator.is_local_main_process:
            saved_query_emb_fp = open(saved_query_emb_path, "w")
        for batch_idx, batch in tqdm(
                enumerate(train_queries_dataloader), disable=not accelerator.is_local_main_process,
                total=len(train_queries_dataloader), desc=f"Getting query vectors ..."):

            for k in list(batch.keys()):
                if k.endswith("_query"):
                    batch[k[:-6]] = batch.pop(k)
            with torch.no_grad():
                embs = get_representation_tensor(query_enc_model(**batch)).contiguous()
            qids = batch["qids"]

            qids, embs = accelerator.gather(qids), accelerator.gather(embs)
            if accelerator.is_local_main_process:
                for qid, emb in zip(qids.cpu().numpy(), embs.detach().cpu().numpy()):
                    qid = int(qid)
                    dict_sparse = sparse_vector_to_dict(
                        emb, vocab_id2token, quantization_factor, dummy_token=tokenizer.unk_token)
                    saved_query_emb_fp.write(str(qid) + "\t" + dict_sparse_to_string(dict_sparse) + os.linesep)
                line_cnt += args.eval_batch_size
                if line_cnt > per_file_query_num_limit:  # reset output file
                    file_cnt += 1
                    saved_query_emb_fp.close()
                    print(f"File: {query_sparse_file_name} saved")
                    query_sparse_file_name = f"train_query_sparse_emb_{file_cnt}.tsv"
                    saved_query_emb_path = os.path.join(work_dir, query_sparse_file_name)
                    saved_query_emb_fp = open(saved_query_emb_path, "w")
                    line_cnt = 0

        if accelerator.is_local_main_process:
            saved_query_emb_fp.close()
        accelerator.wait_for_everyone()

    enc_model.train()
    if query_model is not None:
        query_model.train()

    final_search_result_path = os.path.join(work_dir, f"train_search_result.trec")
    if accelerator.is_local_main_process:
        if not file_exists(final_search_result_path):
            for file_index in range(1, expect_query_file_num + 1):
                saved_query_emb_path = os.path.join(work_dir, f"train_query_sparse_emb_{file_index}.tsv")
                search_result_path = os.path.join(work_dir, f"train_search_result{file_index}.trec")
                if accelerator.is_local_main_process:
                    os.system(
                        f"sh {anserini_path}/target/appassembler/bin/SearchCollection -hits 1000 -parallelism 128 \
                        -index {saved_index_dir} \
                        -topicreader TsvInt -topics {saved_query_emb_path} \
                        -output {search_result_path} -format trec \
                        -impact -pretokenized"
                    )
                # -stemmer none

            # Merge search_result_path
            logger.info("Merging search results")
            os.system(f"rm {final_search_result_path}")
            for file_index in range(1, expect_query_file_num + 1):
                search_result_path = os.path.join(work_dir, f"train_search_result{file_index}.trec")
                os.system(f"more {search_result_path} >> {final_search_result_path}")

    accelerator.wait_for_everyone()

    
    # Convert MS MARCO qrels to TREC qrels.
    org_marco_train_qrels_path = os.path.join(args.data_dir, "msmarco/passage_ranking/qrels.train.tsv")
    marco_dev_qrels_path = os.path.join(work_dir, "qrels.train.trec")
    if accelerator.is_local_main_process:
        if not file_exists(org_marco_train_qrels_path):
            with open(marco_dev_qrels_path, 'w') as fout:
                for line in open(org_marco_train_qrels_path):
                    fout.write(line.replace('\t', ' '))
    accelerator.wait_for_everyone()
    # calculate score will cause segmentation fault
    # logger.info(f"!!!: The train metrics at the end of a stage:")
    # os.system(
    #     f"{anserini_path}/tools/eval/trec_eval.9.0.4/trec_eval -c -M 10 -m recip_rank \
    #     {marco_dev_qrels_path} {final_search_result_path}")
    
    # os.system(
    #     f"{anserini_path}/tools/eval/trec_eval.9.0.4/trec_eval -c -mrecall -mmap \
    #     {marco_dev_qrels_path} {final_search_result_path}")
    # accelerator.wait_for_everyone()
    
    qid2pids = load_trec(final_search_result_path)  # About 9 mins.
    qid2negatives = {}
    for q_index, qid in tqdm(enumerate(qid2pids), desc="Getting negatives ..."):
        qid = int(qid)
        gold_pids = train_queries_dataset.qid2pids[qid]

        negative_pids = []
        for pid in qid2pids[qid]:
            if pid not in gold_pids:
                negative_pids.append(pid)
            if len(negative_pids) == args.num_negatives:
                break
        qid2negatives[qid] = negative_pids

    return qid2negatives

def get_qid2negatives_path(args):
    qid2negatives_path = os.path.join(args.output_dir, f"top_{args.num_negatives}_qid2negatives.pkl")
    if file_exists(qid2negatives_path):
        if not args.overwrite_save_path:
            raise ValueError("Output static hard negative path ({}) already exists. Use --overwrite_save_path to overcome.".format(args.overwrite_save_path))
    return qid2negatives_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_load_type", type=str, default="disk", choices=["disk", "memory"])
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_negatives", type=int, default=200)
    parser.add_argument("--num_proc", type=int, default=4)
    parser.add_argument("--encoder", type=str, default='distilbert')
    parser.add_argument("--overwrite_save_path", action='store_true')
    # set up arguments
    model_param_list = add_model_hyperparameters(parser)
    add_model_configs(parser)
    args = parser.parse_args()
    config, tokenizer = load_config_and_tokenizer(args)
    for param in model_param_list:
        setattr(config, param, getattr(args, param))
    qid2negatives_path = get_qid2negatives_path(args)
    
    # prepare for model
    accelerator = Accelerator(
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=True)])
    prepare_logger(args, accelerator)
    args.per_device_eval_batch_size = args.eval_batch_size // accelerator.num_processes

    if args.encoder == "distilbert":
        encoder_class = DistilBertSpladeEnocder
    elif args.encoder == "bert":
        encoder_class = BertSpladeEnocder
    elif args.encoder == "condenser":
        encoder_class = ConSpladeEnocder
    elif args.encoder == "roberta":
        encoder_class = RobertaSpladeEnocder
    else:
        raise NotImplementedError(args.encoder)

    encoder = encoder_class.from_pretrained(args.model_name_or_path, config=config)
    encoder = accelerator.prepare(encoder)

    embedding_dim = encoder.embedding_dim if hasattr(encoder, "embedding_dim") else len(tokenizer.get_vocab())
    org_vocab = tokenizer.get_vocab()
    vocab_id2token = dict((i, s) for s, i in tokenizer.get_vocab().items())
    if embedding_dim == len(vocab_id2token):
        pass
    elif embedding_dim > len(vocab_id2token):
        for i in range(embedding_dim - len(vocab_id2token)):
            token = f"ex#$!@{i}"
            assert token not in org_vocab
            assert len(vocab_id2token) not in vocab_id2token
            vocab_id2token[len(vocab_id2token)] = token
    else:
        vocab_id2token = {}
        for i in range(embedding_dim):
            token = f"@{i}"
            vocab_id2token[i] = token

    qid2negatives = get_hard_negative_by_retriever(args, encoder, accelerator, vocab_id2token=vocab_id2token, tokenizer=tokenizer, quantization_factor=100,)

    logger.info("Saving hard qid2negative to {}".format(qid2negatives_path))
    with open(qid2negatives_path, "wb") as fp:
        pickle.dump(qid2negatives, fp, protocol=4)


if __name__ == "__main__":
    main()


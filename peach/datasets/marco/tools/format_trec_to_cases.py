import os
from peach.datasets.marco.tools.utils_marco_results import load_trec, load_golden_qrels, load_pid2text, load_qid2text


def format_trec_to_cases(
        trec_path, data_dir, output_path
):
    pid2passage = load_pid2text(os.path.join(data_dir, "msmarco/collection.tsv"))
    qid2query = load_qid2text(os.path.join(data_dir, "msmarco/passage_ranking/dev.query.txt"))
    qid2pids = load_golden_qrels(os.path.join(data_dir, "msmarco/passage_ranking/qrels.dev.tsv"))
    qid2results = load_trec(trec_path)

    col_names = ["qid", "pid", "qrel", "rank", "score", "query", "title<sep>passage"]

    with open(output_path, "w") as fp:
        fp.write("\t".join(col_names) + os.linesep)
        for qid, pid2meta in qid2results.items():
            for pid, meta in pid2meta.items():
                qrel = int(pid in qid2pids[qid])
                rank = meta["rank"]
                score = meta["score"]
                query = qid2query[qid]
                passage = pid2passage[pid]
                fp.write(
                    f"{qid}\t{pid}\t{qrel}\t{rank}\t{score}\t{query}\t{passage}{os.linesep}"
                )


def format_trec_to_cases_limit(
        trec_path, data_dir, output_path,
        keep_num_queries=160, keep_top_n=100,
):
    pid2passage = load_pid2text(os.path.join(data_dir, "msmarco/collection.tsv"))
    qid2query = load_qid2text(os.path.join(data_dir, "msmarco/passage_ranking/dev.query.txt"))
    qid2pids = load_golden_qrels(os.path.join(data_dir, "msmarco/passage_ranking/qrels.dev.tsv"))
    qid2results = load_trec(trec_path)

    col_names = ["qid", "pid", "qrel", "rank", "score", "query", "title<sep>passage"]

    all_qids = list(qid2results.keys())
    stride = len(all_qids) // keep_num_queries
    all_qids = [qid for idx, qid in enumerate(all_qids) if idx % stride == 0]
    print(len(all_qids))
    all_qids = set(all_qids[:keep_num_queries])

    with open(output_path, "w") as fp:
        fp.write("\t".join(col_names) + os.linesep)
        for qid, pid2meta in qid2results.items():
            if qid not in all_qids: continue
            for pid, meta in pid2meta.items():
                qrel = int(pid in qid2pids[qid])
                rank = meta["rank"]
                score = meta["score"]
                query = qid2query[qid]
                passage = pid2passage[pid]
                fp.write(
                    f"{qid}\t{pid}\t{qrel}\t{rank}\t{score}\t{query}\t{passage}{os.linesep}"
                )
                if rank == keep_top_n:
                    break


if __name__ == '__main__':
    # part 1: calculate metrif for trec file
    # calc_metric_for_trec_file(
    #     "/Users/tshen/Downloads/model_retrieval_scores/bm25_official.trec",
    #     "/Users/tshen/Downloads/qrels.dev.tsv",
    # )

    # part 2
    trec_paths = [
        "/home/tshen/Downloads/model_retrieval_scores/bm25_official.trec",
        "/home/tshen/Downloads/model_retrieval_scores/co-stg2.retrieval.trec",
        "/home/tshen/Downloads/model_retrieval_scores/sp-stg2.retrieval.trec",
        "/home/tshen/Downloads/model_retrieval_scores/ensemble_sum_co-stg2_sp-stg2.retrieval.trec",
    ]
    for trec_path in trec_paths:
        format_trec_to_cases_limit(
            trec_path=trec_path,
            data_dir="/relevance2-nfs/shentao/text_corpus/doc_pretrain_corpus",
            output_path=trec_path + ".cases_n6980_t20.tsv",
            keep_num_queries=6980, keep_top_n=20,
        )



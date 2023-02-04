import collections
import pickle
from tqdm import tqdm
import numpy as np
from peach.datasets.marco.tools.utils_marco_results import calculate_metric, load_golden_qrels, \
    load_trec, save_list_to_file, qid2results_to_trec_str_list

def load_kai_pkl(dense_search_result_path):
    """
    dense_1000_scores # [query_num, 1000]
    dense_1000_pids # [query_num, 1000]
    dense_qids # [query_num, ]
    """

    with open(dense_search_result_path, "rb") as fp:
        dense_1000_scores, dense_1000_pids, dense_qids = pickle.load(fp)

    qid2results = dict()  # {qid: {pid: {score: , rank: }}}
    for qid_index, qid in enumerate(dense_qids):
        qid = int(qid)
        if qid not in qid2results:
            qid2results[qid] = dict()
        assert len(dense_1000_pids[qid_index]) == len(dense_1000_scores[qid_index])

        for i in range(len(dense_1000_pids[qid_index])):  # TAO: add support for more than top 1000
            # pid, score, rank
            pid = int(dense_1000_pids[qid_index][i])
            score = float(dense_1000_scores[qid_index][i])
            rank = i + 1
            if pid not in qid2results[qid].keys():
                qid2results[qid][pid] = dict()
            qid2results[qid][pid]['score'] = score
            qid2results[qid][pid]['rank'] = rank

    return qid2results


def official_top1000dev_to_trec_str_list(official_top1000dev_path):
    qid2top1000pids = collections.defaultdict(list)
    qd_pairs = load_tsv(official_top1000dev_path)
    # offsets = get_line_offsets(msmarco_passage_dev_top1000_path, encoding="utf-8")
    for idx, qd_pair in enumerate(qd_pairs):  # grouped by "qid"
        assert len(qd_pair) == 4
        qid, pid = int(qd_pair[0]), int(qd_pair[1])
        qid2top1000pids[qid].append(pid)

    trec_str_list = []
    all_qid_list = list(sorted(qid2top1000pids.keys()))
    print("all_qid_list", len(all_qid_list))
    for qid in all_qid_list:
        results = [(pid, (idx+1), 1.0/(idx+1),) for idx, pid in enumerate(qid2top1000pids[qid])]
        trec_str_list.extend(
            [f"{qid} Q0 {res[0]} {res[1]} {res[2]} BM25_Official" for res in results][:TOP_K]
        )
    print("OFFICIAL_BM25", len(trec_str_list))
    return trec_str_list

if __name__ == '__main__':
    # part 1: official
    # save_list_to_file(
    #     official_top1000dev_to_trec_str_list("/Users/tshen/Downloads/top1000.dev"),
    #     "/Users/tshen/Downloads/model_retrieval_scores/bm25_official.trec"
    # )

    # # part 2: official
    # qid2results = load_kai_pkl("/Users/drogokhal/Downloads/model_retrieval_scores/co-stg1.retrieval.raw.pkl")
    # trec_str_list = qid2results_to_trec_str_list(qid2results, "Dense_Cocon_Stage1")
    # save_list_to_file(trec_str_list, "/Users/drogokhal/Downloads/model_retrieval_scores/co-stg1.retrieval.trec")
    #
    # qid2results = load_kai_pkl("/Users/drogokhal/Downloads/model_retrieval_scores/co-stg2.retrieval.raw.pkl")
    # trec_str_list = qid2results_to_trec_str_list(qid2results, "Dense_Cocon_Stage2")
    # save_list_to_file(trec_str_list, "/Users/drogokhal/Downloads/model_retrieval_scores/co-stg2.retrieval.trec")

    qid2results = load_kai_pkl("./output/co-stg2/search_result.pkl")
    trec_str_list = qid2results_to_trec_str_list(qid2results, "Dense_Cocon_Stage2")
    save_list_to_file(trec_str_list, "./output/co-stg2/search_result.trec")








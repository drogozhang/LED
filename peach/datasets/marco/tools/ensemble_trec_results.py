import collections
from tqdm import tqdm
import numpy as np
from peach.datasets.marco.tools.utils_marco_results import calculate_metric, load_golden_qrels, \
    load_trec, save_list_to_file, qid2results_to_trec_str_list

def max_min_norm_scores(a):
    a = np.array(a)
    assert len(a.shape) == 1
    return (a - np.min(a)) / (np.max(a) - np.min(a)).tolist()


def simple_aggregation(score_list):
    return sum(score_list)  # / len(score_list)


def trec_ensemble(
        trec_result_list, score_norm_method=max_min_norm_scores,
        score_aggretation_method=simple_aggregation,
):
    all_qids = list(trec_result_list[0].keys())
    all_qids.sort()
    all_qids = all_qids
    qid2results = dict()
    for qid in tqdm(all_qids, desc="trec_ensemble"):

        all_ps_pairs = [[(pid, meta["score"]) for pid, meta in res_dict[qid].items()]
                       for res_dict in trec_result_list]
        all_pid_scores = [list(zip(*e)) for e in all_ps_pairs]
        all_pids = [e[0] for e in all_pid_scores]
        all_norm_scores = [score_norm_method(e[1]) for e in all_pid_scores]

        pid2score_list = collections.defaultdict(list)
        for pids, norm_scores in zip(all_pids, all_norm_scores):
            for pid, score in zip(pids, norm_scores):
                pid2score_list[pid].append(score)
        pid_ensemble_score_list = []
        for pid, score_list in pid2score_list.items():
            pid_ensemble_score_list.append(
                (
                    pid,
                    score_aggretation_method(score_list)
                )
            )
        pid_ensemble_score_list.sort(key=lambda e:e[1], reverse=True)
        pid_ensemble_score_list = [  # pid, rank, score
            (pid, idx+1, score) for idx, (pid, score) in enumerate(pid_ensemble_score_list)]

        out_pid2meta = dict()
        for pid, rank, score in  pid_ensemble_score_list:
            out_pid2meta[pid] = {"rank": rank, "score": score,}
        qid2results[qid] = out_pid2meta
    return qid2results



if __name__ == '__main__':
    trec_path_list = [
        # "/Users/drogokhal/Downloads/model_retrieval_scores/co-stg2.retrieval.trec",
        # "/Users/drogokhal/Downloads/model_retrieval_scores/sp-stg2.retrieval.trec",
        # "/Users/drogokhal/Downloads/model_retrieval_scores/co-stg2.retrieval.trec",
        # "/Users/drogokhal/Downloads/model_retrieval_scores/sp-stg2.retrieval.trec",
        # "/Users/drogokhal/Downloads/model_retrieval_scores/anserini-bm25.retrieval.trec",
        # "output/entangle/stg1-co-bs16nn7-lr5e6-ep3-sp-stg2-dst-rank-1.2-co-stg1/dense_retrieval/search_result.trec",
        "./output/entangle/stg1-co-bs16nn24-lr5e6-ep3-sp-stg2-dst-rank-1.2-co-stg1/dense_retrieval/search_result.trec",
        # "output/sp-stg2/search_result.trec"
        "/data/shentao/runtime/unicocon-stage2_archive4/unicocon-Dcls-stage2-ST-pos-othernegs-LMD0024-LWd1.0s1.0-lr8e-6ep2bs12ns150nn30-SD24-EVAL/sparse_retrieval/dev_search_result_hits2048.trec"
    ]
    # output_path = "/Users/drogokhal/Downloads/model_retrieval_scores/ensemble_sum_co-stg2_sp-stg2.retrieval.trec"
    # output_path = "./output/ensemble/ensemble_sum_sp-stg2_cp-stg1-dst.retrieval.trec"
    # output_path = "./output/ensemble/unicocon-Dcls-stage2-ST-pos-othernegs-LMD0024-LWd1.0s1.0-lr8e-6ep2bs12ns150nn30-SD24-EVAL-top1000.retrieval.trec"

    # output_path = "./output/ensemble/unicocon-Dcls-stage2-ST-pos-othernegs-LMD0024-LWd1.0s1.0-lr8e-6ep2bs12ns150nn30-SD24-------stg1-co-bs16nn24-lr5e6-ep3-sp-stg2-dst-rank-1.2-co-stg1.retrieval.trec"
    trec_result_list = [load_trec(p) for p in trec_path_list]

    qid2results = trec_ensemble(
        trec_result_list,
    )
    # trec_str_list = qid2results_to_trec_str_list(qid2results, "Ensemble_co-stg2_sp-stg2")
    # trec_str_list = qid2results_to_trec_str_list(qid2results, "Ensemble_co-stg2_anserini-bm25")
    # save_list_to_file(trec_str_list, output_path)
    # print(calculate_metric(qid2results, load_golden_qrels("/Users/drogokhal/Downloads/qrels.dev.tsv",)))
    print(calculate_metric(qid2results, load_golden_qrels("/data/shentao/corpus/msmarco/passage_ranking/qrels.dev.tsv",)))


# if __name__ == '__main__': # only computing metrics
#     print("Runnning...")
#     ensemble_path = "/Users/drogokhal/Downloads/model_retrieval_scores/ensemble_sum_co-stg2_sp-stg2.retrieval.trec"
#     qid2results = load_trec(ensemble_path)
#     print(calculate_metric(qid2results, load_golden_qrels("/Users/drogokhal/Downloads/qrels.dev.tsv")))

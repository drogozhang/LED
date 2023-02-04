import collections
from tqdm import tqdm
import numpy as np
from proj_marco_misc.utils import calculate_metrics_for_qid2results, save_qid2results_to_trec, load_trec
# from utils import calculate_metrics_for_qid2results, save_qid2results_to_trec, load_trec
import argparse

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
        all_ps_pairs = [[(pid, score) for pid, score in res_dict[qid].items()]
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
        pid_ensemble_score_list = [  # pid, score
            (pid, score) for pid, score in pid_ensemble_score_list]
        
        pid2enscore = dict((pid, scores) for pid, scores in pid_ensemble_score_list)
        qid2results[qid] = pid2enscore

        # out_pid2meta = dict()
        # for pid, rank, score in  pid_ensemble_score_list:
        #     out_pid2meta[pid] = {"rank": rank, "score": score,}
        # qid2results[qid] = out_pid2meta
    return qid2results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--trec_paths", type=str, default="co-stg2.retrieval.trec;sp-stg2.retrieval.trec")
    parser.add_argument("--qrels_path", type=str, default="qrels.dev.tsv")
    parser.add_argument("--output_path", type=str, default=None)
    args = parser.parse_args()
    trec_path_list = args.trec_paths.split(";")
    trec_result_list = [load_trec(p) for p in trec_path_list]

    qid2results = trec_ensemble(
        trec_result_list,
    )

    if args.output_path is not None:
        save_qid2results_to_trec(
            qid2results, top_k=1000, source_name="Ensemble", save_to_file=args.output_path)

    eval_metrics = calculate_metrics_for_qid2results(qid2results, args.qrels_path)

    print(eval_metrics)
    

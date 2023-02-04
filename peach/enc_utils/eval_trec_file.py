import datasets
from tqdm import tqdm
from peach.datasets.marco.tools.utils_marco_results import calculate_metric, load_trec


def evaluate_trec_file(trec_file_path, qrels_file_path):
    trec_results = load_trec(trec_file_path)

    metric_ranking = datasets.load_metric("peach/metrics/ranking_v2.py")
    metric_ranking.qids_list = []

    for qid in tqdm(trec_results, desc="Calculating metrics ..."):
        qid = int(qid)
        res = trec_results[qid]
        pred_pids = [pid for pid, _ in res.items()]
        top_scores = [meta["score"] for pid, meta in res.items()]
        metric_ranking.add_batch(predictions=top_scores, references=pred_pids)
        metric_ranking.qids_list.extend([qid] * len(res))
    eval_metrics = metric_ranking.compute(
        group_labels=metric_ranking.qids_list,
        qrels_path=qrels_file_path
    )
    return eval_metrics


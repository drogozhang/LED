import datasets
import numpy as np
import math
import collections
import torch
MaxMRRRank = 10
ROUND_DIGIT=6

import pytrec_eval

import logging
from typing import List, Dict, Union, Tuple

def calc_ndcg(qrels: Dict[str, Dict[str, int]],
        results: Dict[str, Dict[str, float]],
        k_values: List[int]) -> Tuple[Dict[str, float]]:
    ndcg = {}
    for k in k_values:
        ndcg[f"NDCG@{k}"] = 0.0
    ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {ndcg_string, })
    scores = evaluator.evaluate(results)
    for query_id in scores.keys():
        for k in k_values:
            ndcg[f"NDCG@{k}"] += scores[query_id]["ndcg_cut_" + str(k)]
    for k in k_values:
        ndcg[f"NDCG@{k}"] = round(ndcg[f"NDCG@{k}"] / len(scores), ROUND_DIGIT)
    return ndcg


def calc_recall_official(qrels: Dict[str, Dict[str, int]],
        results: Dict[str, Dict[str, float]],
        k_values: List[int]) -> Tuple[Dict[str, float]]:
    recall = {}
    for k in k_values:
        recall[f"recall_official@{k}"] = 0.0
    recall_string = "recall." + ",".join([str(k) for k in k_values])
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {recall_string, })
    scores = evaluator.evaluate(results)
    for query_id in scores.keys():
        for k in k_values:
            recall[f"recall_official@{k}"] += scores[query_id]["recall_" + str(k)]
    for k in k_values:
        recall[f"recall_official@{k}"] = round(recall[f"recall_official@{k}"] / len(scores), ROUND_DIGIT)
    return recall


def calc_mrr_official(qrels: Dict[str, Dict[str, int]],
        results: Dict[str, Dict[str, float]],
        k_values: List[int]) -> Tuple[Dict[str, float]]:
    MRR = dict()

    top_hits = dict()
    k_max = max(k_values)
    for query_id, doc_scores in results.items():
        top_hits[query_id] = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)[0:k_max]

    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"recip_rank", })
    for k in k_values:
        local_results = dict((query_id, dict((k, v) for k, v in ths[:k])) for query_id, ths in top_hits.items())
        scores = evaluator.evaluate(local_results)
        all_scores = [scores[query_id]["recip_rank"] for query_id in scores]
        MRR[f"MRR_official@{k}"] = round(sum(all_scores) / len(all_scores), ROUND_DIGIT)
    return MRR


def calc_mrr(qrels: Dict[str, Dict[str, int]],
        results: Dict[str, Dict[str, float]],
        k_values: List[int]) -> Tuple[Dict[str, float]]:
    MRR = {}

    for k in k_values:
        MRR[f"MRR@{k}"] = 0.0

    k_max, top_hits = max(k_values), {}
    # logging.info("\n")

    for query_id, doc_scores in results.items():
        top_hits[query_id] = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)[0:k_max]

    for query_id in top_hits:
        query_relevant_docs = set([doc_id for doc_id in qrels[query_id] if qrels[query_id][doc_id] > 0])
        for k in k_values:
            for rank, hit in enumerate(top_hits[query_id][0:k]):
                if hit[0] in query_relevant_docs:
                    MRR[f"MRR@{k}"] += 1.0 / (rank + 1)
                    break

    for k in k_values:
        MRR[f"MRR@{k}"] = round(MRR[f"MRR@{k}"] / len(qrels), ROUND_DIGIT)
        # logging.info("MRR@{}: {:.4f}".format(k, MRR[f"MRR@{k}"]))

    return MRR


def calc_recall_rocketqa(
        qrels: Dict[str, Dict[str, int]],
        results: Dict[str, Dict[str, float]],
        k_values: List[int]) -> Tuple[Dict[str, float]]:
    capped_recall = {}

    for k in k_values:
        capped_recall[f"recall@{k}"] = 0.0

    k_max = max(k_values)
    # logging.info("\n")

    for query_id, doc_scores in results.items():
        top_hits = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)[0:k_max]
        query_relevant_docs = [doc_id for doc_id in qrels[query_id] if qrels[query_id][doc_id] > 0]
        for k in k_values:
            retrieved_docs = [row[0] for row in top_hits[0:k] if qrels[query_id].get(row[0], 0) > 0]
            capped_recall[f"recall@{k}"] += int(len(retrieved_docs) > 0)
            # denominator = min(len(query_relevant_docs), k)
            # capped_recall[f"recall@{k}"] += (len(retrieved_docs) / denominator)

    for k in k_values:
        capped_recall[f"recall@{k}"] = round(capped_recall[f"recall@{k}"] / len(qrels), ROUND_DIGIT)
        # logging.info("recall@{}: {:.4f}".format(k, capped_recall[f"recall@{k}"]))

    return capped_recall


def calc_recall_cap(qrels: Dict[str, Dict[str, int]],
               results: Dict[str, Dict[str, float]],
               k_values: List[int]) -> Tuple[Dict[str, float]]:
    capped_recall = {}

    for k in k_values:
        capped_recall[f"recall@{k}"] = 0.0

    k_max = max(k_values)
    # logging.info("\n")

    for query_id, doc_scores in results.items():
        top_hits = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)[0:k_max]
        query_relevant_docs = [doc_id for doc_id in qrels[query_id] if qrels[query_id][doc_id] > 0]
        for k in k_values:
            retrieved_docs = [row[0] for row in top_hits[0:k] if qrels[query_id].get(row[0], 0) > 0]
            denominator = min(len(query_relevant_docs), k)
            capped_recall[f"recall@{k}"] += (len(retrieved_docs) / denominator)

    for k in k_values:
        capped_recall[f"recall@{k}"] = round(capped_recall[f"recall@{k}"] / len(qrels), ROUND_DIGIT)
        # logging.info("recall@{}: {:.4f}".format(k, capped_recall[f"recall@{k}"]))

    return capped_recall


def calc_hole(qrels: Dict[str, Dict[str, int]],
         results: Dict[str, Dict[str, float]],
         k_values: List[int]) -> Tuple[Dict[str, float]]:
    Hole = {}

    for k in k_values:
        Hole[f"Hole@{k}"] = 0.0

    annotated_corpus = set()
    for _, docs in qrels.items():
        for doc_id, score in docs.items():
            annotated_corpus.add(doc_id)

    k_max = max(k_values)
    # logging.info("\n")

    for _, scores in results.items():
        top_hits = sorted(scores.items(), key=lambda item: item[1], reverse=True)[0:k_max]
        for k in k_values:
            hole_docs = [row[0] for row in top_hits[0:k] if row[0] not in annotated_corpus]
            Hole[f"Hole@{k}"] += len(hole_docs) / k

    for k in k_values:
        Hole[f"Hole@{k}"] = round(Hole[f"Hole@{k}"] / len(qrels), ROUND_DIGIT)
        # logging.info("Hole@{}: {:.4f}".format(k, Hole[f"Hole@{k}"]))

    return Hole


def calc_top_k_accuracy(
        qrels: Dict[str, Dict[str, int]],
        results: Dict[str, Dict[str, float]],
        k_values: List[int]) -> Tuple[Dict[str, float]]:
    top_k_acc = {}

    for k in k_values:
        top_k_acc[f"Accuracy@{k}"] = 0.0

    k_max, top_hits = max(k_values), {}
    # logging.info("\n")

    for query_id, doc_scores in results.items():
        top_hits[query_id] = [item[0] for item in
                              sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)[0:k_max]]

    for query_id in top_hits:
        query_relevant_docs = set([doc_id for doc_id in qrels[query_id] if qrels[query_id][doc_id] > 0])
        for k in k_values:
            for relevant_doc_id in query_relevant_docs:
                if relevant_doc_id in top_hits[query_id][0:k]:
                    top_k_acc[f"Accuracy@{k}"] += 1.0
                    break

    for k in k_values:
        top_k_acc[f"Accuracy@{k}"] = round(top_k_acc[f"Accuracy@{k}"] / len(qrels), ROUND_DIGIT)
        # logging.info("Accuracy@{}: {:.4f}".format(k, top_k_acc[f"Accuracy@{k}"]))

    return top_k_acc

class MetricRankingV2(datasets.Metric):
    def _info(self) -> datasets.MetricInfo:
        return datasets.MetricInfo(
            description="",
            citation="",
            features=datasets.Features(
                {
                    "predictions": datasets.Value("float64"),  # logits [n_example,]
                    "references": datasets.Value("int64"),  # probs [n_example,]  # passage id !!
                    # "group_labels": datasets.Value("int64"),
                }
            ),
        )

    def _compute(
            self, *, predictions=None, references=None, group_labels=None,
            qrels_path=None, num_examples=None, **kwargs):
        if num_examples is not None:
            predictions = predictions[:num_examples]
            references = references[:num_examples]
            group_labels = group_labels[:num_examples]
        # read qrels
        qrels = dict()
        idx = 0
        with open(qrels_path) as fp:
            for line in fp:
                data = line.strip().split("\t")
                qsid, psid, gold_score = str(int(data[0])), str(int(data[2])), int(data[3])
                if qsid not in qrels:
                    qrels[qsid] = dict()
                qrels[qsid][psid] = gold_score
                idx += 1
        # print(qrels_path, len(qrels), idx, sum(len(v) for v in qrels.values()))

        # re-org results to dict
        results = dict()
        for prediction, reference, group_label in zip(predictions, references, group_labels):
            if isinstance(group_labels, torch.Tensor):
                qsid = f"{group_label.item()}"
            else:
                qsid = f"{int(group_label)}"
            if qsid not in results:
                results[qsid] = dict()
            psid = f"{reference}"
            results[qsid][psid] = float(prediction)
        if len(qrels) != len(results):
            print(f"Warning: qrels (len is {len(qrels)}) is not equal to results (len is {len(results)})")
        results = dict((k, v) for k, v in results.items() if k in qrels)
        # calculate ndcg, mrr, recall
        res_metrics = {"QueriesRanked": len(results)}
        res_metrics.update(calc_mrr(qrels, results, [10, 100]))
        res_metrics.update(calc_mrr_official(qrels, results, [10, 100]))
        res_metrics.update(calc_recall_rocketqa(qrels, results, [1, 5, 10, 20, 50, 100, 500, 1000, 5000]))
        res_metrics.update(calc_recall_official(qrels, results, [1, 5, 10, 20, 50, 100, 500, 1000, 5000]))
        res_metrics.update(calc_ndcg(qrels, results, [10, 100]))
        return res_metrics

import datasets
import numpy as np
import math
import collections
import torch
MaxMRRRank = 10


class MetricRanking(datasets.Metric):
    def _info(self) -> datasets.MetricInfo:
        return datasets.MetricInfo(
            description="",
            citation="",
            features=datasets.Features(
                {
                    "predictions": datasets.Value("float64"),  # logits [n_example,]
                    "references": datasets.Value("int64"),  # probs [n_example,]
                    # "group_labels": datasets.Value("int64"),
                }
            ),
        )

    def _compute(self, *, predictions=None, references=None, group_labels=None, num_examples=None, **kwargs):
        # group by query
        if num_examples is not None:
            predictions = predictions[:num_examples]
            references = references[:num_examples]
            group_labels = group_labels[:num_examples]

        assert len(predictions) == len(references)

        qid2predictions = collections.defaultdict(list)
        for prediction, reference, group_label in zip(predictions, references, group_labels):
            if isinstance(group_labels, torch.Tensor):
                qid2predictions[group_label.item()].append((float(prediction), int(reference), ))
            else:
                qid2predictions[int(group_label)].append((float(prediction), int(reference),))

        MRR = 0.
        ranking = []
        recall_q_top1 = set()
        recall_q_top20 = set()
        recall_q_top50 = set()
        recall_q_top100 = set()
        recall_q_all = set()

        for qid in qid2predictions:
            qid2predictions[qid].sort(key=lambda e: e[0], reverse=True)
            predictions = qid2predictions[qid]
            ranking.append(0)
            for i in range(0, min(MaxMRRRank, len(predictions))):
                pred, ref = qid2predictions[qid][i]
                if ref == 1:
                    MRR += 1.0 / (i + 1)
                    ranking.pop()
                    ranking.append(i + 1)
                    break
            for i, (pred, ref) in enumerate(predictions):
                if ref == 1:
                    recall_q_all.add(qid)
                    if i < 100:
                        recall_q_top100.add(qid)
                    if i < 50:
                        recall_q_top50.add(qid)
                    if i < 20:
                        recall_q_top20.add(qid)
                    if i == 0:
                        recall_q_top1.add(qid)
                    break
        if len(ranking) == 0:
            raise IOError("No matching QIDs found. Are you sure you are scoring the evaluation set?")

        num_queries = len(qid2predictions)
        MRR = MRR / num_queries
        recall_top1 = len(recall_q_top1) * 1.0 / num_queries
        recall_top20 = len(recall_q_top20) * 1.0 / num_queries
        recall_top50 = len(recall_q_top50) * 1.0 / num_queries
        recall_top100 = len(recall_q_top100) * 1.0 / num_queries
        recall_all = len(recall_q_all) * 1.0 / num_queries

        all_scores = {
            'MRR@10': MRR,
            'recall@1': recall_top1,
            'recall@20': recall_top20,
            'recall@50': recall_top50,
            'recall@100': recall_top100,
            'recall@all': recall_all,
            'QueriesRanked': num_queries,
        }

        # all_scores[MRR @10] = MRR
        # all_scores["recall@1"] = recall_top1
        # all_scores["recall@50"] = recall_top50
        # all_scores["recall@all"] = recall_all
        # all_scores['QueriesRanked'] = len(qids_to_ranked_candidate_passages)
        return all_scores





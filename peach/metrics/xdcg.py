
import datasets
import numpy as np
import math
import collections

XDCG_DISCOUNT = 0.6

def compute_ndcg_xdcg(query_dict):
    """Computes XDCG."""
    xdcg_at_1_sum = {}
    xdcg_at_3_sum = {}

    xdcg_at_1_sum["all"] = 0.0
    xdcg_at_3_sum["all"] = 0.0

    query_count = {}
    query_count["all"] = 0

    query_gain = []

    for market,query_label_score_list in query_dict.items():
        query_count["all"] += len(query_label_score_list)
        query_count[market] = len(query_label_score_list)

        for query,label_score_list in query_label_score_list.items():
            label_score_list.sort(key=lambda tup: tup[1], reverse=True)
            xdcg = compute_xdcg(label_score_list, depth=3)

            xdcg_at_1_sum["all"] += xdcg[0]
            xdcg_at_3_sum["all"] += xdcg[2]

            query_gain.append(market+'\t'+query+'\t'+str(xdcg[0])+'\t'+str(xdcg[2]/1.96)+'\n')

            if market not in xdcg_at_1_sum:
                xdcg_at_1_sum[market] = xdcg[0]
                xdcg_at_3_sum[market] = xdcg[2]
            else:
                xdcg_at_1_sum[market] += xdcg[0]
                xdcg_at_3_sum[market] += xdcg[2]

    for market,value in xdcg_at_1_sum.items():
        xdcg_at_1_sum[market] /= query_count[market]

    for market,value in xdcg_at_3_sum.items():
        xdcg_at_3_sum[market] /= query_count[market]
        xdcg_at_3_sum[market] /= 1.96

    return xdcg_at_1_sum, xdcg_at_3_sum,query_count,query_gain

def compute_xdcg(docs_label_score, depth=1):
    """Compute and return XDCG given a list ranked documents."""
    if depth <= 0:
        raise Exception("Invalid depth for xdcg calculation.")

    xdcg = np.zeros(depth)
    num_docs = len(docs_label_score)

    for i in range(depth):
        # current gain
        if i < num_docs:
            xdcg_label = float(docs_label_score[i][0]) * 25
            xdcg[i] = xdcg_label * math.pow(XDCG_DISCOUNT, i)
        # add previous gain
        if i > 0:
            xdcg[i] += xdcg[i-1]

    return xdcg

# xdcg_at_1, xdcg_at_3, query_count, query_gain = compute_ndcg_xdcg(market_query_dict)

class MetricXDCG(datasets.Metric):
    def _info(self) -> datasets.MetricInfo:
        return datasets.MetricInfo(
            description="",
            citation="",
            features=datasets.Features(
                {
                    "predictions": datasets.Value("float64"),  # logits [n_example,]
                    "references": datasets.Value("float64"),  # probs [n_example,]
                    # "group_labels": datasets.Value("int64"),
                }
            ),
        )

    def _compute(self, *, predictions=None, references=None, group_labels=None, **kwargs):
        market_query_dict = {
            "en": collections.defaultdict(list),
        }
        for pred, ref, gl in zip(predictions, references, group_labels):
            market_query_dict["en"][str(gl)].append((float(ref), float(pred)))

        xdcg_at_1, xdcg_at_3, query_count, query_gain = compute_ndcg_xdcg(market_query_dict)

        out_dict = {
            "xdcg_at_1": xdcg_at_1["en"],
            "xdcg_at_3": xdcg_at_3["en"],
        }
        return out_dict









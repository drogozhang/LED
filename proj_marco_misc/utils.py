import logging
import os
import sys
import collections
import csv

csv.field_size_limit(sys.maxsize)


def load_tsv(input_file, quotechar=None):
    """Reads a tab separated value file."""
    with open(input_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
        lines = []
        for line in reader:
            if sys.version_info[0] == 2:
                line = list(unicode(cell, 'utf-8') for cell in line)
            lines.append(line)
        return lines


def save_list_to_file(str_list, file_path, use_basename=False):
    with open(file_path, "w", encoding="utf-8") as fp:
        for path_str in str_list:
            fp.write(os.path.basename(path_str) if use_basename else path_str)
            fp.write(os.linesep)


def load_trec(trec_file):
    logging.info(f"loading trec {trec_file}")
    qid2results = dict()  # {qid: {pid: {score: , rank: }}}
    with open(trec_file) as fp:
        for idx_line, line in enumerate(fp):
            try:
                qid, _, pid, rank, score, _method = line.strip().split(" ")
            except ValueError:
                print(f"Please check line {idx_line} in {trec_file}")
                raise ValueError
            qid, pid, rank, score = int(qid), int(pid), int(rank), float(score)
            # if rank <= TOP_K:  # only count TOP K  TAO: add support
            if qid not in qid2results:  # initialize
                qid2results[qid] = collections.OrderedDict()
            qid2results[qid][pid] = score
    return qid2results


def save_qid2results_to_trec(
        qid2results, top_k=1000, source_name="NoSourceName", save_to_file=None, ):
    trec_str_list = []
    all_qid_list = list(sorted(qid2results.keys()))
    for qid in all_qid_list:
        results = [(pid, score) for pid, score in qid2results[qid].items()]
        results.sort(key=lambda e: e[1], reverse=True)
        cur_results = [f"{qid} Q0 {pid} {idx+1} {score} {source_name}" for idx, (pid, score) in enumerate(results)]
        if len(cur_results) < top_k:
            logging.info(f"WARN: qid-{qid} only has {len(cur_results)} passage results, less than {top_k}!")
        trec_str_list.extend(cur_results[:top_k])
    logging.info(f"Trec from {source_name}, num of line is {len(trec_str_list)}")
    if save_to_file is not None:
        logging.info(f"save to {save_to_file}")
        save_list_to_file(trec_str_list, save_to_file)
    return trec_str_list

def transform_qid2results_to_qid2hn(qid2results, top_k, qrels_path):
    qrels = load_tsv(os.path.join(qrels_path))
    qid2pos_pids = collections.defaultdict(set)
    for qrel in qrels:
        assert len(qrel) == 4
        qid, pid = int(qrel[0]), int(qrel[2])  # todo: judge if > 0
        qid2pos_pids[qid].add(pid)

    qid2negatives = dict()
    all_qid_list = list(sorted(qid2results.keys()))
    for qid in all_qid_list:
        qid = int(qid)
        pos_pids = qid2pos_pids[qid]
        results = [(pid, score) for pid, score in qid2results[qid].items()]
        results.sort(key=lambda e: e[1], reverse=True)
        negs = [int(pid) for pid, s in results if int(pid) not in pos_pids]
        if len(negs) < top_k:
            logging.info(f"WARN: qid-{qid} only has {len(negs)} passage results, less than {top_k}!")
        qid2negatives[qid] = negs[:top_k]
    return qid2negatives

def calculate_metrics_for_qid2results(qid2results, qrels_path):
    import datasets
    from tqdm import tqdm
    metric_ranking = datasets.load_metric("peach/metrics/ranking_v2.py")
    metric_ranking.qids_list = []
    for qid, results in tqdm(qid2results.items(), desc="Calculating metrics ..."):
        pids, scores = zip(*[(int(pid), score) for pid, score in results.items()])
        metric_ranking.add_batch(predictions=scores, references=pids)
        metric_ranking.qids_list.extend([int(qid)] * len(results))
    eval_metrics = metric_ranking.compute(
        group_labels=metric_ranking.qids_list,
        qrels_path=qrels_path,
    )
    return eval_metrics

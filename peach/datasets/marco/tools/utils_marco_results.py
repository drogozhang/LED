import os
import sys
import collections
import csv

csv.field_size_limit(sys.maxsize)

TOP_K = 1000

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


def qid2results_to_trec_str_list(qid2results, source_name):
    # 2 Q0 2022782 20 209860.000000 Anserini
    # qid Q0
    trec_str_list = []
    all_qid_list = list(sorted(qid2results.keys()))
    for qid in all_qid_list:
        results = [(pid, meta["rank"], meta["score"],) for pid, meta in qid2results[qid].items()]
        results.sort(key=lambda e: e[1])
        trec_str_list.extend(
            [f"{qid} Q0 {res[0]} {res[1]} {res[2]} {source_name}" for res in results][:TOP_K]
        )
    print(source_name, len(trec_str_list))
    return trec_str_list


def load_trec(trec_file, top_k=1000):
    print(f"loading trec {trec_file}")
    qid2results = dict()  # {qid: {pid: {score: , rank: }}}
    with open(trec_file) as fp:
        for line in fp:
            qid, _, pid, rank, score, _method = line.strip().split(" ")
            qid, pid, rank, score = int(qid), int(pid), int(rank), float(score)
            if rank <= top_k:  # only count TOP K  TAO: add support
                if qid not in qid2results:  # initialize
                    qid2results[qid] = collections.OrderedDict()
                qid2results[qid][pid] = {
                    'score': score,
                    'rank': rank,}
        # line = fp.readline()
        # while line:
                # if pid not in qid2results[qid]:
                #     qid2results[qid][pid] = dict()  # for score and rank
                # qid2results[qid][pid]['score'] = score
                # qid2results[qid][pid]['rank'] = rank
            # line = fp.readline()
    return qid2results


# msmarco/passage_ranking/qrels.dev.tsv
def load_golden_qrels(qrels_file_path):
    qrels = load_tsv(qrels_file_path)
    qid2pids = collections.defaultdict(set)
    for qrel in qrels:
        assert len(qrel) == 4
        qid2pids[int(qrel[0])].add(int(qrel[2]))
    return qid2pids

# msmarco/collection.tsv
def load_pid2text(collection_path):
    pid_title = load_tsv(collection_path + ".title.tsv")
    pid2title = dict((int(pid), title) for pid, title in pid_title)

    pid2passage = dict()
    with open(collection_path, encoding="utf-8") as fp:
        for idx_line, line in enumerate(fp):
            passage_id, passage = line.strip("\n").split("\t")
            pid2passage[int(passage_id)] = passage

    for pid in list(pid2passage.keys()):
        pid2passage[pid] = pid2title[pid] + "<sep>" + pid2passage[pid]
    return pid2passage

# msmarco/passage_ranking/dev.query.txt
def load_qid2text(query_path):
    queries = load_tsv(query_path)
    qid2query = dict((int(qid), query_text) for qid, query_text in queries)
    return qid2query


def calculate_metric(qid2results, golden_qid2pids):
    import datasets
    from tqdm import tqdm
    import numpy as np

    metric_ranking = datasets.load_metric("peach/metrics/ranking.py")
    metric_ranking.qids_list = []

    for qid in tqdm(qid2results, desc="Calculating metrics ..."):
        qid = int(qid)
        res = qid2results[qid]
        gold_pids = golden_qid2pids[qid]
        pred_pids = [pid for pid, _ in res.items()]
        top_references = np.array([int(pid in gold_pids) for pid in pred_pids], dtype="int64")
        top_scores = [meta["score"] for pid, meta in res.items()]
        metric_ranking.add_batch(predictions=top_scores, references=top_references)
        metric_ranking.qids_list.extend([qid] * len(res))

    eval_metrics = metric_ranking.compute(group_labels=metric_ranking.qids_list)
    return eval_metrics


def calc_metric_for_trec_file(trec_path, gold_qrel_path):
    qid2pids = load_golden_qrels(gold_qrel_path)
    qid2results = load_trec(trec_path)
    metric = calculate_metric(qid2results, qid2pids)
    print(trec_path, metric)



# if __name__ == "__main__":
#     # qid2results = load_trec("/home/tao/data/msmarco/passage_ranking/dev.trec")

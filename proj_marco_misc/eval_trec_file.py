from proj_marco_misc.utils import calculate_metrics_for_qid2results, save_qid2results_to_trec, \
    load_trec
from tqdm import tqdm
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--trec_path", type=str, required=True)
    parser.add_argument("--qrels_path", type=str, required=True)
    args = parser.parse_args()

    qid2results = load_trec(args.trec_path)
    eval_metrics = calculate_metrics_for_qid2results(qid2results, args.qrels_path)
    print(eval_metrics)

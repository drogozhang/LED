import argparse, os
from peach.common import file_exists, get_line_offsets, save_json, load_json, \
    load_list_from_file, load_tsv, http_get, save_pickle, load_pickle


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    data_dir = args.data_dir
    output_dir = args.output_dir

    qid_pids_list = load_tsv(
        os.path.join(data_dir, "msmarco/passage_ranking/train.negatives.tsv"))
    qid2pids = {}
    for qid, pids in qid_pids_list:
        neg_pids = [int(pid) for pid in pids.split(",")]
        qid2pids[int(qid)] = neg_pids

    save_pickle(qid2pids, os.path.join(output_dir, "bm25-off.pkl"), protocol=4)


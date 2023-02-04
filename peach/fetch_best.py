import glob
import os
from peach.common import dir_exists, file_exists
import argparse

def _load_first_line(_path):
    with open(_path) as fp:
        return fp.readline().strip()


def fetch_best_results(dir_pattern, best_file="best_eval_results.txt"):
    res_list = []
    for _dir in glob.glob(dir_pattern):
        _file_path = os.path.join(_dir, best_file)
        # print(_file_path)
        if dir_exists(_dir) and file_exists(_file_path):
            _result = _load_first_line(_file_path)
            _metric = float(_result.split(",")[0])
            meta_data = (_dir.split("/")[-1], _result)
            res_list.append((_metric, meta_data))

    res_list = list(sorted(res_list, key=lambda _e: _e[0], reverse=True))

    print("*"*20)
    for _metric, _meta_data in res_list:
        print("{}    {}".format(_metric, _meta_data))
    print("*" * 20)
    print("Total {} and best is {}".format(len(res_list), res_list[0][0] if len(res_list) > 0 else None))
    print("*" * 20)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--dir_pattern", type=str, required=True)
    arg_parser.add_argument("--best_filename", type=str, default="best_eval_results.txt")

    args = arg_parser.parse_args()
    fetch_best_results(args.dir_pattern, best_file=args.best_filename)





















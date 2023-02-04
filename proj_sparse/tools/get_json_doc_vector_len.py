from peach.common import get_statistics_for_num_list
import argparse
import json
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument("--path", type=str, required=True)
parser.add_argument("--key", type=str, default="vector")

args = parser.parse_args()

num_list = []
with open(args.path) as fp:
    for idx, line in tqdm(enumerate(fp)):
        data = json.loads(line)
        num_list.append(len(data[args.key]))

        if idx > 100000:
            break

print(get_statistics_for_num_list(num_list))



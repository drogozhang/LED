import argparse
import logging
import math
import os
import gc
import random
from dataclasses import dataclass
from typing import Optional, Union
import numpy as np
import datasets
import torch
from datasets import load_dataset, load_metric
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm
from collections import defaultdict
import torch.distributed as dist

import transformers
import accelerate
from accelerate import Accelerator
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AdamW,
    AutoConfig,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    set_seed,
    BatchEncoding
)
from transformers.file_utils import PaddingStrategy

from torch.utils.data import TensorDataset, Dataset
import datasets
from torch.utils.tensorboard import SummaryWriter

from abc import abstractmethod, ABCMeta
from contextlib import nullcontext

USER_HOME = os.getenv("HOME")

POS_INF = 10000
NEG_INF = -POS_INF
VERY_SMAIL_FT = 1.0/10000


logger = logging.getLogger()   # __name__
# You should update this to your particular problem to have better documentation of `model_type`
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


class NewDataset(Dataset):
    def preprocess_function(self, example):
        return {}

    def collate_fn(self, features):
        if not isinstance(features[0], (dict, BatchEncoding)):
            features = [vars(f) for f in features]

        first = features[0]
        keys = set(first.keys())
        batch = {}

        def pad_feature_tensor(feat_list, pad_num=0, confirm_no_pad=False):
            # fn1: traverse feat_list
            # find the dim
            feat, dim = feat_list[0], 0
            while isinstance(feat, list):
                feat = feat[0]
                dim += 1
            # type information
            if isinstance(feat, int):
                tensor_dtype = torch.long
            else:
                pad_num = float(pad_num)
                tensor_dtype = torch.float
            # max seq for every dim
            def get_max_lens(flist, mseq, lvl=-1):
                if lvl > -1:
                    mseq[lvl] = max(mseq[lvl], len(flist))
                if isinstance(flist[0], list):
                    for sub_flist in flist:
                        get_max_lens(sub_flist, mseq, lvl=lvl+1)
            # padding
            def pad_all_dims(flist, mseq, lvl=-1):
                if (len(flist) == 0 and lvl < len(mseq)-1) or (len(flist) > 0 and isinstance(flist[0], list)):
                    if lvl > -1:
                        flist.extend([[] for _ in range(mseq[lvl] - len(flist))])
                    for sub_flist in flist:
                        pad_all_dims(sub_flist, mseq, lvl=lvl+1)
                else:
                    assert lvl > -1
                    flist.extend([pad_num for _ in range(mseq[lvl] - len(flist))])
            if not confirm_no_pad and dim != 0:
                max_seq = [0] * dim
                get_max_lens(feat_list, max_seq)
                pad_all_dims(feat_list, max_seq)
            return torch.tensor(feat_list, dtype=tensor_dtype)
        # common and other
        # seq_labels = set(k for k in keys if k.endswith("seq_labels"))
        label_keys = set(k for k in keys if k.endswith("labels"))  # - seq_labels
        # common_keys = ["input_ids", "token_type_ids", "attention_mask",]
        for k in keys:
            if k in label_keys:
                batch[k] = pad_feature_tensor([f[k] for f in features], pad_num=-100)  # confirm_no_pad=True
            else:
                pad_num = 0
                if k == "input_ids" and self.tokenizer.pad_token is not None:
                    pad_num = self.tokenizer.pad_token_id
                batch[k] = pad_feature_tensor([f[k] for f in features], pad_num=pad_num)
        return batch

    def __init__(self, args, raw_datasets, data_type, tokenizer, accelerator, column_names=None):
        self.args = args
        self.raw_datasets = raw_datasets
        self.data_type = data_type
        self.tokenizer = tokenizer
        self.accelerator = accelerator
        self.column_names = column_names

        assert data_type in ["train", "dev", "test"]

        self.raw_set = self.raw_datasets[self.data_type] if self.data_type != "dev" else self.raw_datasets["validation"]

        # pre-filter
        self.raw_set = self.raw_set.filter(function=self.pre_filtering_fn)

        self.padding = "max_length" if self.args.pad_to_max_length else False

        if self.column_names is None:
            self.column_names = self.raw_set.column_names

        # First we tokenize all the texts.
        self.raw_set = self.raw_set.map(
            self.preprocess_function,
            batched=False,
            # remove_columns=self.column_names
            num_proc=None if self.args.num_proc <= 1 or self.data_type != "train" else self.args.num_proc
        )
        # set format for __get_item__

        self.raw_set.set_format(
            type=None, columns=[cn for cn in self.numeric_columns if cn in self.raw_set.column_names])

    def __getitem__(self, *args, **kwargs):
        return self.raw_set.__getitem__(*args, **kwargs)

    def __len__(self):
        return len(self.raw_set)

    def pre_filtering_fn(self, example):
        return True

    def get_metric(self):
        pass

    @property
    def key_metric_name(self):
        return None

    @property
    def test_has_label(self):
        return False

    @property
    def numeric_columns(self):
        return ['input_ids', 'token_type_ids', 'attention_mask', 'labels', ]


def get_collate_fn(tokenizer):

    def collate_fn(features):
        if not isinstance(features[0], (dict, BatchEncoding)):
            features = [vars(f) for f in features]

        first = features[0]
        keys = set(first.keys())
        batch = {}

        def pad_feature_tensor(feat_list, pad_num=0, confirm_no_pad=False):
            # fn1: traverse feat_list
            # find the dim
            feat, dim = feat_list[0], 0
            while isinstance(feat, list):
                feat = feat[0]
                dim += 1
            # type information
            if isinstance(feat, int):
                tensor_dtype = torch.long
            else:
                pad_num = float(pad_num)
                tensor_dtype = torch.float
            # max seq for every dim
            def get_max_lens(flist, mseq, lvl=-1):
                if lvl > -1:
                    mseq[lvl] = max(mseq[lvl], len(flist))
                if isinstance(flist[0], list):
                    for sub_flist in flist:
                        get_max_lens(sub_flist, mseq, lvl=lvl+1)
            # padding
            def pad_all_dims(flist, mseq, lvl=-1):
                if (len(flist) == 0 and lvl < len(mseq)-1) or (len(flist) > 0 and isinstance(flist[0], list)):
                    if lvl > -1:
                        flist.extend([[] for _ in range(mseq[lvl] - len(flist))])
                    for sub_flist in flist:
                        pad_all_dims(sub_flist, mseq, lvl=lvl+1)
                else:
                    assert lvl > -1
                    flist.extend([pad_num for _ in range(mseq[lvl] - len(flist))])
            if not confirm_no_pad and dim != 0:
                max_seq = [0] * dim
                get_max_lens(feat_list, max_seq)
                pad_all_dims(feat_list, max_seq)
            return torch.tensor(feat_list, dtype=tensor_dtype)
        # common and other
        # seq_labels = set(k for k in keys if k.endswith("seq_labels"))
        label_keys = set(k for k in keys if (k.startswith("labels") or k.endswith("labels")))  # - seq_labels
        # common_keys = ["input_ids", "token_type_ids", "attention_mask",]
        for k in keys:
            if k in label_keys:
                batch[k] = pad_feature_tensor([f[k] for f in features], pad_num=-100)  # confirm_no_pad=True
            else:
                pad_num = 0
                if (k.startswith("input_ids") or k.endswith("input_ids")) and tokenizer.pad_token is not None:
                    pad_num = tokenizer.pad_token_id
                batch[k] = pad_feature_tensor([f[k] for f in features], pad_num=pad_num)
        return batch
    return collate_fn


def define_hparams_training(parser):
    # parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_prediction", action='store_true',
                        help="Whether to run eval on the test set and save predictions")
    parser.add_argument("--num_proc", type=int, default=1, help="max cpu cores",)

    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")

    parser.add_argument('--logging_steps', type=int, default=100,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=-1,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument('--eval_steps', type=int, default=-1,
                        help="Eval model every X updates steps. if X > 0")

    # ====== Copy from Huggingface example ======
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--dev_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--test_file", type=str, default=None, help="A csv or a json file containing the test data."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument("--train_batch_size", default=32, type=int, help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=12, type=int, help="Total batch size for eval.")
    # parser.add_argument(
    #     "--per_device_train_batch_size",
    #     type=int,
    #     default=8,
    #     help="Batch size (per device) for the training dataloader.",
    # )
    # parser.add_argument(
    #     "--per_device_eval_batch_size",
    #     type=int,
    #     default=8,
    #     help="Batch size (per device) for the evaluation dataloader.",
    # )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--task_learning_rate",
        type=float,
        default=0.0,
        help="Initial learning rate (after the potential warmup period) to use for task modules.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--adam_betas", default=None, type=str,
                        help='betas for Adam optimizer')
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")

    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--warmup_proportion", default=0.05, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")

    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Activate debug mode and run training only with a subset of data.",
    )
    # args = parser.parse_args()
    # if args.output_dir is not None:
    #     os.makedirs(args.output_dir, exist_ok=True)

    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local rank for DDP",
    )

    return parser

def define_hparams_generation(parser):
    parser.add_argument(
        "--do_sample", action="store_true",
    )
    parser.add_argument(
        "--num_beams", default=1, type=int, required=False, help="beams for beam search"
    )
    parser.add_argument(
        "--temperature",
        default=1.0,
        type=float,
        required=False,
        help="temperature for sampling",
    )

    parser.add_argument(
        "--top_k", default=50, type=int, required=False, help="k for top k sampling"
    )
    # Using p=0.9 by default
    parser.add_argument(
        "--top_p", default=1.0, type=float, required=False, help="p for nucleus sampling"
    )
    parser.add_argument(
        "--num_return_sequences", default=5, type=int, required=False, help="k"
    )

    return parser

def check_hparams_generation(args):
    pass  # todo


def setup_prerequisite(args):
    # 1. Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator(
        fp16=True,
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=True)])
    # accelerator = Accelerator()

    # 2. output dir  # todo: how to resume training
    if accelerator.is_local_main_process:
        if os.path.exists(args.output_dir) and os.listdir(
                args.output_dir) and args.do_train and not args.overwrite_output_dir:
            raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)  # Create output directory if needed
    accelerator.wait_for_everyone()

    # We can further explore the usage of `split_batches=True`

    assert args.train_batch_size % (accelerator.num_processes * args.gradient_accumulation_steps) == 0, \
        f"batch_size: {args.train_batch_size}, num_processes: {accelerator.num_processes}, accumulation_steps: {args.gradient_accumulation_steps}"
    args.per_device_train_batch_size = args.train_batch_size // (accelerator.num_processes * args.gradient_accumulation_steps)
    args.per_device_eval_batch_size = args.eval_batch_size  # only eval at local
    
    # 3. Make one log on every process with the configuration for debugging.
    # logging.basicConfig(
    #     # filename=os.path.join(args.output_dir, "logging.txt"),
    #     format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    #     datefmt="%m/%d/%Y %H:%M:%S",
    #     level=logging.INFO,
    # )
    if accelerator.is_local_main_process:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S")
        fh = logging.FileHandler(os.path.join(args.output_dir, "logging.txt"))  # use FileHandler to file
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        ch = logging.StreamHandler()  # use StreamHandler to console
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)  # add to Handler
        logger.addHandler(fh)
    else:
        logger.setLevel(logging.WARNING)
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S")
        fh = logging.FileHandler(os.path.join(args.output_dir, f"logging_subprocess-{dist.get_rank()}.txt"))  # use FileHandler to file
        fh.setLevel(logging.WARNING)
        fh.setFormatter(formatter)
        ch = logging.StreamHandler()  # use StreamHandler to console
        ch.setLevel(logging.WARNING)
        ch.setFormatter(formatter)
        logger.addHandler(ch)  # add to Handler
        logger.addHandler(fh)


    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    # logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    # if accelerator.is_local_main_process:
    #     datasets.utils.logging.set_verbosity_warning()
    #     transformers.utils.logging.set_verbosity_info()
    # else:
    #     datasets.utils.logging.set_verbosity_error()
    #     transformers.utils.logging.set_verbosity_error()
    logger.info(accelerator.state)

    # 4. If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed + max(0, accelerator.local_process_index))

    return accelerator


def load_config_and_tokenizer(args, config_kwargs=None, tk_kwargs=None):
    config_kwargs = config_kwargs or dict()
    tk_kwargs = tk_kwargs or dict()
    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name, **config_kwargs)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer, **tk_kwargs)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer, **tk_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    return config, tokenizer


def init_new_tokens_embeddings(model, tokenizer, new_tokens):
    """
    Initialize each attribute embedding (e.g. <very-bad>) with the bag of words of its words (vec(very) + vec(bad))
    """
    model.resize_token_embeddings(len(tokenizer))

    if new_tokens is None or len(new_tokens) == 0:
        return

    embeddings = model.get_input_embeddings()
    unk_token_id = tokenizer.unk_token_id # tokenizer.convert_tokens_to_ids(tokenizer.unk_token)

    for word in set(new_tokens):
        index = tokenizer.convert_tokens_to_ids(word)
        strip_word = word
        if word.startswith("[|") or word.startswith("<|"):
            strip_word = strip_word[2:-2]
        elif word.startswith("[") or word.startswith("<"):
            strip_word = strip_word[1:-1]
        else:
            continue  # because
        strip_word = " ".join(strip_word.split("-"))
        other_words = tokenizer.tokenize(strip_word)
        other_indices = tokenizer.convert_tokens_to_ids(other_words)
        other_indices = [i for i in other_indices if unk_token_id is None or i != unk_token_id]
        if len(other_indices) == 0:
            continue
        elif len(other_indices) == 1:
            vec = embeddings.weight.data[other_indices[0], :]
        else:
            vec = torch.sum(
                torch.stack([embeddings.weight.data[i, :] for i in other_indices]))
        logger.info(
            f"Setting w[{index}] = {word} to the average of {other_words} ({other_indices})"
        )
        embeddings.weight.data[index, :] = vec


def add_new_tokens_to_tokenizer(tokenizer, special_tokens_dict=None, new_tokens=None, ):  # model=None, init_emb=False
    if special_tokens_dict is not None and len(special_tokens_dict)>0:
        tokenizer.add_special_tokens(special_tokens_dict)

    if new_tokens is not None and len(new_tokens) > 0:
        tokenizer.add_tokens(new_tokens, special_tokens=False)

    # if model is not None:
    #     model.resize_token_embeddings(len(tokenizer))
    #     if init_emb and (new_tokens is not None and len(new_tokens) > 0):
    #         init_new_tokens_embeddings(model, tokenizer, new_tokens)


def load_raw_datasets(args):
    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
    else:
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.dev_file is not None:
            data_files["validation"] = args.dev_file
        if args.test_file is not None:
            data_files["test"] = args.test_file
        extension = args.train_file.split(".")[-1]
        if extension == "jsonl":
            extension = "json"
        if extension == "tsv":
            extension = "csv"
        if extension == "txt":
            extension = "text"
        raw_datasets = load_dataset(extension, data_files=data_files)

    if hasattr(args, "data_scale") and isinstance(args.data_scale, float):
        num_train_ex = len(raw_datasets["train"])
        num_keep = int(num_train_ex * args.data_scale)
        keep_indices = list(range(num_train_ex))
        random.shuffle(keep_indices)
        keep_indices = keep_indices[:num_keep]
        raw_datasets["train"] = raw_datasets["train"].select(keep_indices)

    # Trim a number of training examples
    if args.debug:
        for split in raw_datasets.keys():
            raw_datasets[split] = raw_datasets[split].select(range(200))
    return raw_datasets


def setup_train_dataloader(args, train_dataset, accelerator, **kwargs):
    # todo
    train_dataloader = DataLoader(
        train_dataset, shuffle=True,
        collate_fn=train_dataset.collate_fn if hasattr(train_dataset, "collate_fn") else get_collate_fn(train_dataset.tokenizer),
        batch_size=args.per_device_train_batch_size,
        num_workers=args.num_proc,
    )
    train_dataloader = accelerator.prepare(train_dataloader)
    return train_dataloader


def setup_opt(args, model, accelerator, num_batches_per_epoch):
    # Optimizer
    # Split weights in two groups, one with weight decay and the other not. # todo: distinguish bert params with others
    no_decay = ["bias", "LayerNorm.weight"]

    if hasattr(args, "task_learning_rate") and args.task_learning_rate > 0.:
        param_pattern = model.base_model_prefix + "."
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if
                        n.startswith(param_pattern) and (not any(nd in n for nd in no_decay))],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if
                        n.startswith(param_pattern) and any(nd in n for nd in no_decay)],
             'weight_decay': 0.0},
            {'params': [p for n, p in model.named_parameters() if
                        (not n.startswith(param_pattern)) and (not any(nd in n for nd in no_decay))],
             'weight_decay': args.weight_decay, 'lr': args.task_learning_rate},
            {'params': [p for n, p in model.named_parameters() if
                        (not n.startswith(param_pattern)) and any(nd in n for nd in no_decay)],
             'weight_decay': 0.0, 'lr': args.task_learning_rate},
        ]
    else:
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

    # customized
    if args.adam_betas is not None:
        adam_betas = tuple(float(_f) for _f in args.adam_betas.split(","))
        assert len(adam_betas) == 2
    else:
        adam_betas = (0.9, 0.999)

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate,
                      betas=adam_betas, eps=args.adam_epsilon)

    # Use the device given by the `accelerator` object.
    # device = accelerator.device
    # model.to(device)

    # Prepare everything with our `accelerator`.
    model, optimizer = accelerator.prepare(
        model, optimizer  # train_dataloader, eval_dataloader
    )

    # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
    # shorter in multiprocess)

    # Scheduler and math around the number of training steps.
    args.num_update_steps_per_epoch = math.ceil(num_batches_per_epoch / args.gradient_accumulation_steps)
    if args.max_train_steps is None:  # max_train_steps has 1st priority
        args.max_train_steps = args.num_train_epochs * args.num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / args.num_update_steps_per_epoch)

    if args.warmup_proportion > VERY_SMAIL_FT:  # warmup_proportion has the 1st priority
        args.num_warmup_steps = int(args.max_train_steps * args.warmup_proportion)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps ,
        num_training_steps=args.max_train_steps,
    )

    # Train!
    # args.total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    return model, optimizer, lr_scheduler


def logging_berfore_training(args, train_dataset):
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {args.train_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")


def update_wrt_loss(args, accelerator, model, optimizer, loss):
    loss = loss / args.gradient_accumulation_steps
    accelerator.backward(loss)
    return loss


def model_update_wrt_gradient(args, accelerator, model, optimizer, lr_scheduler):
    # gradient clip
    if args.max_grad_norm > VERY_SMAIL_FT:
        accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad()


def setup_eval_dataloader(args, eval_dataset, accelerator, use_accelerator=False, **kwargs):
    if not use_accelerator:
        eval_dataloader = DataLoader(
            eval_dataset, collate_fn=eval_dataset.collate_fn if hasattr(eval_dataset, "collate_fn") else get_collate_fn(eval_dataset.tokenizer),
            batch_size=args.per_device_eval_batch_size,
            num_workers=args.num_proc,
        )
    else:
        # sampler = SequentialDistributedSampler(eval_dataset, batch_size=args.per_device_eval_batch_size,)
        eval_dataloader = DataLoader(
            eval_dataset,
            collate_fn=eval_dataset.collate_fn if hasattr(eval_dataset, "collate_fn") else get_collate_fn(eval_dataset.tokenizer),
            batch_size=args.per_device_eval_batch_size,
            num_workers=args.num_proc,
        )
        eval_dataloader = accelerator.prepare(eval_dataloader)
    # Note there is no `accelerator.prepare` for eval_dataloader
    return eval_dataloader


def save_model_with_default_name(
        args, accelerator, output_dir, model, tokenizer=None, args_to_save=None, wait_for_everyone=True,
        save_specified_module=None, **kwargs,
):
    if accelerator.is_local_main_process:

        # Create output directory if needed
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        logger.info("Saving model checkpoint to %s", output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        # model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
        # model_to_save.save_pretrained(output_dir)

        unwrapped_model = accelerator.unwrap_model(model)
        if save_specified_module is None:
            unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
        else:
            module_to_save = getattr(unwrapped_model, save_specified_module)
            module_to_save.save_pretrained(output_dir, save_function=accelerator.save)

        if tokenizer is not None:
            tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        if args_to_save is not None:
            torch.save(args_to_save, os.path.join(output_dir, 'training_args.bin'))

    if wait_for_everyone:
        accelerator.wait_for_everyone()



# =========
def train(args, train_dataset, model, accelerator, tokenizer, eval_dataset=None, eval_fn=None):
    if accelerator.is_local_main_process:
        tb_writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "tensorboard"))
    else:
        tb_writer = None

    train_dataloader = setup_train_dataloader(args, train_dataset, accelerator)
    model, optimizer, lr_scheduler = setup_opt(args, model, accelerator, len(train_dataloader))

    logging_berfore_training(args, train_dataset)

    # Metrics
    # metric = load_metric("accuracy")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    global_step = 0
    step_loss = 0.
    step_loss_dict = defaultdict(float)
    best_metric = NEG_INF
    ma_dict = MovingAverageDict()
    model.zero_grad()

    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            model.train()
            outputs = model(**batch)
            # calculate loss
            update_wrt_loss(args, accelerator, model, optimizer, outputs["loss"])
            # update
            for key in outputs:
                if key.endswith("loss"):
                    step_loss_dict[key] += outputs[key].item() / args.gradient_accumulation_steps

            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                model_update_wrt_gradient(args, accelerator, model, optimizer, lr_scheduler)
                progress_bar.update(1)
                global_step += 1
                # update loss for logging
                if tb_writer is not None:  # local main process
                    ma_dict(step_loss_dict)
                    for key, loss_val in step_loss_dict.items():
                        tb_writer.add_scalar(f"training-{key}", loss_val, global_step)
                step_loss_dict = defaultdict(float)

                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    if accelerator.is_local_main_process:
                        logging.info(ma_dict.get_val_str())

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    save_model_with_default_name(args, accelerator, args.output_dir, model, tokenizer, args)

                if (eval_dataset is not None and eval_fn is not None) and args.eval_steps > 0 and global_step % args.eval_steps == 0:
                    key_metric, eval_metrics = eval_fn(args, eval_dataset, model, accelerator, global_step=global_step, tb_writer=tb_writer)
                    if key_metric > best_metric:
                        best_metric = key_metric
                        save_model_with_default_name(args, accelerator, args.output_dir, model, tokenizer, args_to_save=args)

            if global_step >= args.max_train_steps:
                break
        # evaluation each epoch or last epoch
        if (accelerator.is_local_main_process and eval_dataset is not None and eval_fn is not None) and \
                (global_step >= args.max_train_steps or args.eval_steps <= 0):
            key_metric, eval_metrics = eval_fn(args, eval_dataset, model, accelerator, global_step=global_step, tb_writer=tb_writer)
            if key_metric > best_metric:
                best_metric = key_metric
                save_model_with_default_name(args, accelerator, args.output_dir, model, tokenizer, args_to_save=args)


def standard_training_and_eval_procedure(
        args, accelerator, config, tokenizer, raw_datasets,
        model_class, dataset_class, eval_fn,
        **kwargs
):

    # ====== data pre-processing ======
    train_dataset = dataset_class(args, raw_datasets, "train", tokenizer, accelerator, column_names=None)
    dev_dataset = dataset_class(args, raw_datasets, "dev", tokenizer, accelerator,
                                  column_names=train_dataset.column_names)
    if "test" in raw_datasets:
        test_dataset = dataset_class(args, raw_datasets, "test", tokenizer, accelerator,
                                       column_names=train_dataset.column_names)
    else:
        test_dataset = None

    if args.model_name_or_path:
        model = model_class.from_pretrained(
            args.model_name_or_path, config=config,
        )
    else:
        logger.info("Training new model from scratch")
        model = model_class.from_config(config)
    model.resize_token_embeddings(len(tokenizer))

    if args.do_train:
        train(args, train_dataset, model, accelerator, tokenizer,
              eval_dataset=dev_dataset if args.do_eval else None,
              eval_fn=eval_fn if args.do_eval else None,
              )

    if args.do_eval or args.do_prediction:
        if args.do_train:
            model = model_class.from_pretrained(
                args.output_dir, config=config,
                # from_tf=bool(".ckpt" in args.model_name_or_path),
            )
        else:
            pass
        model = accelerator.prepare(model)

        save_best_metric, save_best_str = None, ""
        if args.do_eval:
            best_dev_result, _ = eval_fn(args, dev_dataset, model, accelerator, global_step=None, save_prediction=True)
            # args, eval_dataset, model, accelerator, global_step=global_step, tb_writter=tb_writer
            save_best_metric = save_best_metric if save_best_metric is not None else best_dev_result
            save_best_str += f"best_dev_result: {best_dev_result}, "
        if args.do_prediction and test_dataset is not None:
            best_test_result, _ = eval_fn(args, test_dataset, model, accelerator, global_step=None, save_prediction=True)
            save_best_metric = save_best_metric if save_best_metric is not None else best_test_result
            save_best_str += f"best_test_result: {best_test_result}, "

        with open(os.path.join(args.output_dir, "best_eval_results.txt"), "w") as fp:
            fp.write(f"{save_best_metric}, {save_best_str}{os.linesep}")


def evaluate_cls(args, eval_dataset, model, accelerator, global_step=None, tb_writer=None):
    accelerator.wait_for_everyone()
    if not accelerator.is_local_main_process:
        return

    model.eval()
    eval_dataloader = setup_eval_dataloader(args, eval_dataset, accelerator, use_accelerator=False)
    logger.info(f"Evaluation for {eval_dataset.data_type}:")
    # Metrics
    metric = eval_dataset.get_metric()

    for step, batch in enumerate(eval_dataloader):
        batch = dict((k, v.to(accelerator.device)) for k, v in batch.items())
        with torch.no_grad():
            outputs = model(**batch)
        metric.add_batch(
            predictions=outputs.logits.argmax(dim=-1),
            references=batch["labels"],
        )

    if (not eval_dataset.data_type == "test") or eval_dataset.test_has_label:
        eval_metric = metric.compute()
        logger.info(f"step {global_step}: {eval_metric}")
        return eval_metric[eval_dataset.key_metric_name], eval_metric
    else:
        return 0., dict()

# ==========
class MovingAverageDict(object):
    def __init__(self, decay=0.99):
        self.decay = decay
        self.ma_dict = {}

    def __call__(self, value_dict):
        for key, val in value_dict.items():
            if isinstance(val, (np.float32, np.float64, np.float16)) or \
                    (isinstance(val, np.ndarray) and val.dtype == "float32" and val.ndim == 0):
                val = float(val)

            if isinstance(val, float):
                if key not in self.ma_dict:
                    self.ma_dict[key] = MovingAverage()
                self.ma_dict[key](val)

    def get_val_dict(self):
        dict_return = {}
        for key, ma_obj in self.ma_dict.items():
            dict_return[key] = ma_obj.value
        return dict_return

    def get_val_str(self):
        val_dict = self.get_val_dict()
        # sort
        sorted_list = list(sorted(val_dict.items(), key=lambda item: item[0]))
        str_return = ""
        for key, val in sorted_list:
            if len(str_return) > 0:
                str_return += ", "
            str_return += "%s: %.4f" % (key, val)
        return str_return


class MovingAverage(object):
    def __init__(self, decay=0.99):
        self.decay = decay
        self.value = None

    def __call__(self, new_val):
        if self.value is None:
            self.value = new_val
        else:
            self.value = self.decay * self.value + (1. - self.decay) * new_val
        return self.value


def get_truncated_seqlen_list(seq_list, budge_len, ratio_list=None):
    ratio_list = ratio_list or ([1.] * len(seq_list))
    assert len(seq_list) > 0
    assert len(seq_list) == len(ratio_list)
    assert budge_len >= len(seq_list)

    lens_np = np.array([len(_e) for _e in seq_list], dtype="int64")
    ratios_np = np.array(ratio_list, dtype="float32")

    while sum(lens_np) > budge_len:
        lens_np[np.argmax(lens_np/ratios_np)] -= 1

    return list(lens_np)


class CustomArgs(object):
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)



# =========
# from: https://github.com/huggingface/transformers/blob/447808c85f0e6d6b0aeeb07214942bf1e578f9d2/src/transformers/trainer_pt_utils.py
class SequentialDistributedSampler(torch.utils.data.sampler.Sampler):
    """
    Distributed Sampler that subsamples indicies sequentially,
    making it easier to collate all results at the end.
    Even though we only use this sampler for eval and predict (no training),
    which means that the model params won't have to be synced (i.e. will not hang
    for synchronization even if varied number of forward passes), we still add extra
    samples to the sampler to make it evenly divisible (like in `DistributedSampler`)
    to make it easy to `gather` or `reduce` resulting tensors at the end of the loop.
    """
    def __init__(self, dataset, batch_size, rank=None, num_replicas=None):
        super().__init__(dataset)
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.batch_size = batch_size
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.batch_size / self.num_replicas)) * self.batch_size
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        # add extra samples to make it evenly divisible
        indices += [indices[-1]] * (self.total_size - len(indices))
        # subsample
        indices = indices[self.rank * self.num_samples : (self.rank + 1) * self.num_samples]
        return iter(indices)

    def __len__(self):
        return self.num_samples


def free_memory():
    gc.collect()
    torch.cuda.empty_cache()


def free_model(model, accelerator=None):
    if accelerator is not None:
        accelerator.free_memory()
    del model
    free_memory()



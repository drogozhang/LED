from ast import Str
import json
import tqdm

from peach.base import *

from peach.datasets.marco.dataset_marco_passages import DatasetMarcoPassagesRanking
from peach.datasets.marco.dataset_marco_eval import DatasetRerank, DatasetCustomRerank

from peach.enc_utils.eval_functions import evaluate_encoder_reranking
from peach.enc_utils.eval_dense import evaluate_dense_retreival
from peach.enc_utils.hn_gen_dense import get_hard_negative_by_dense_retrieval
from peach.enc_utils.general import get_representation_tensor

from peach.enc_utils.enc_learners import LearnerMixin
from transformers import AutoModel

import torch
import torch.nn as nn

from peach.enc_utils.sim_metric import Similarity
from peach.enc_utils.general import preproc_inputs


def add_training_hyperparameters(parser):
    parser.add_argument("--apply_distill", action="store_true")
    pass


class DenseLearner(LearnerMixin):
    def __init__(self, model_args, config, tokenizer, encoder, query_encoder=None):
        super().__init__(model_args, config, tokenizer, encoder, query_encoder, )
        self.sim_fct = Similarity(metric="dot")

    def forward(
            self,
            input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None,
            input_ids_query=None, attention_mask_query=None, token_type_ids_query=None, position_ids_query=None,
            distill_labels=None, training_progress=None,
            training_mode=None,
            **kwargs,
    ):

        if training_mode is None:  # if not training, work like encoder.
            return get_representation_tensor(self.encoder(input_ids, attention_mask, )).contiguous()

        dict_for_meta = {}

        # input reshape
        (high_dim_flag, num_docs), (org_doc_shape, org_query_shape), \
        (input_ids, attention_mask, token_type_ids, position_ids,), \
        (input_ids_query, attention_mask_query, token_type_ids_query, position_ids_query,) = preproc_inputs(
            input_ids, attention_mask, token_type_ids, position_ids,
            input_ids_query, attention_mask_query, token_type_ids_query, position_ids_query,)
        bsz = org_doc_shape[0]

        # encoding
        doc_outputs = self.encoding_doc(input_ids, attention_mask, return_dict=True)
        query_outputs = self.encoding_query(input_ids_query, attention_mask_query, return_dict=True)

        emb_doc, emb_query = get_representation_tensor(doc_outputs), get_representation_tensor(query_outputs)
        emb_dim = emb_doc.shape[-1]

        dict_for_loss = {}
        # calculate similarity and losses
        ga_emb_doc, ga_emb_query = self.gather_tensor(emb_doc), self.gather_tensor(emb_query)

        # if dist.is_initialized():
        dict_for_meta["dense_xentropy_loss_weight"] = float(self.my_world_size)

        loss_fct = nn.CrossEntropyLoss()
        # KZ 3/30/2022 in-batch negatives
        target = torch.arange(ga_emb_query.shape[0], device=ga_emb_query.device, dtype=torch.long) * num_docs

        similarities = self.sim_fct(ga_emb_query, ga_emb_doc.view(-1, emb_dim))
        dict_for_loss["dense_xentropy_loss"] = loss_fct(similarities, target)

        # distillation
        if self.model_args.apply_distill and distill_labels is not None:
            bi_scores = self.sim_fct.forward_qd_pair(
                emb_query.unsqueeze(1), emb_doc.view(bsz, num_docs, -1), )  # bs,nd
            dict_for_loss["distill_ce_loss"] = nn.KLDivLoss()(
                torch.log_softmax(bi_scores, dim=-1), torch.softmax(distill_labels, dim=-1))

        loss = 0.
        for k in dict_for_loss:
            if k + "_weight" in dict_for_meta:
                loss += dict_for_meta[k + "_weight"] * dict_for_loss[k]
            else:
                loss += dict_for_loss[k]
        dict_for_loss["loss"] = loss

        dict_for_meta.update(dict_for_loss)
        return dict_for_meta


def train(args, train_dataset, model, accelerator, tokenizer, eval_dataset=None, eval_fn=None):
    if accelerator.is_local_main_process:
        tb_writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "tensorboard"))
    else:
        tb_writer = None

    train_dataloader = setup_train_dataloader(args, train_dataset, accelerator)
    model, optimizer, lr_scheduler = setup_opt(args, model, accelerator, len(train_dataloader))

    logging_berfore_training(args, train_dataset)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    global_step = 0
    step_loss = 0.
    step_loss_dict = defaultdict(float)
    best_metric = NEG_INF
    ma_dict = MovingAverageDict()
    model.train()
    model.zero_grad()

    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            # print(dist.get_rank(), train_dataloader["input_ids"].shape[0])
            step += 1  # fix for accumulation
            sync_context = model.no_sync if accelerator.distributed_type != accelerate.DistributedType.NO and \
                                            step % args.gradient_accumulation_steps == 0 else nullcontext

            with sync_context():  # disable DDP sync for accumulation step
                outputs = model(training_mode="retrieval_finetune", **batch)
                update_wrt_loss(args, accelerator, model, optimizer, outputs["loss"])
            # update
            for key in outputs:
                if key.endswith("loss"):
                    step_loss_dict[key] += outputs[key].item() / args.gradient_accumulation_steps

            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader):

                model_update_wrt_gradient(args, accelerator, model, optimizer, lr_scheduler)

                # update loss for logging
                if tb_writer is not None:  # local main process
                    ma_dict(step_loss_dict)
                    for key, loss_val in step_loss_dict.items():
                        tb_writer.add_scalar(f"training-{key}", loss_val, global_step)
                    for key, elem in outputs.items():
                        if (key not in step_loss_dict) and isinstance(elem, (int, float)):
                            tb_writer.add_scalar(f"training-meta-{key}", elem, global_step)
                step_loss_dict = defaultdict(float)

                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    if accelerator.is_local_main_process:
                        logging.info(f"Log at step-{global_step}: {ma_dict.get_val_str()}")

                # assert args.save_steps > 0, "save_steps should be larger than 0 when no dev"
                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    save_model_with_default_name(
                        args, accelerator, args.output_dir, model, tokenizer, args, save_specified_module="encoder",)

                # DDP for eval
                if eval_fn is not None and args.eval_steps > 0 and (
                        global_step % args.eval_steps == 0 or global_step == args.max_train_steps-1):
                    # todo: unwrap model
                    if accelerator.is_local_main_process:
                        key_metric, eval_metrics = eval_fn(
                            args, eval_dataset, model, accelerator, global_step=global_step,
                            tb_writer=tb_writer, tokenizer=tokenizer, key_metric_name="MRR@10",
                            similarity_metric=None, query_encoder=None,
                            use_accelerator=False,
                        )
                        if key_metric >= best_metric:  # always false in sub process
                            best_metric = key_metric
                            save_model_with_default_name(
                                args, accelerator, args.output_dir, model, tokenizer, args_to_save=args,
                                wait_for_everyone=False, save_specified_module="encoder",
                                # do not wait other subprocesses during saving in main process
                            )
                    accelerator.wait_for_everyone()  # other subprocesses must wait the main process

                progress_bar.update(1)
                global_step += 1  # update global step after determine whether eval

                if global_step >= args.max_train_steps:
                    break
        save_model_with_default_name(
            args, accelerator, os.path.join(args.output_dir, "last_checkpoint"), model, tokenizer, args, save_specified_module="encoder",)
    save_model_with_default_name(
        args, accelerator, os.path.join(args.output_dir, "last_checkpoint"), model, tokenizer, args, save_specified_module="encoder",)
    model.zero_grad()
    model.eval()
    accelerator.wait_for_everyone()

def main():
    parser = argparse.ArgumentParser()
    # add task specific hyparam
    #
    define_hparams_training(parser)

    parser.add_argument("--data_load_type", type=str, default="disk", choices=["disk", "memory"])
    parser.add_argument("--data_dir", type=str)  # princeton-nlp/sup-simcse-bert-base-uncased
    parser.add_argument("--num_negatives", type=int, default=7)
    parser.add_argument("--num_dev", type=int, default=500)

    parser.add_argument("--ce_score_margin", type=float, default=3.0)
    parser.add_argument("--num_negs_per_system", type=int, default=8)
    parser.add_argument("--negs_sources", type=str, default=None)
    parser.add_argument("--custom_hn_dir", type=str)
    
    parser.add_argument("--no_title", action="store_true")
    # parser.add_argument("--encoder", type=str, default="distilbert", )  # todo: enable

    parser.add_argument("--eval_reranking_source", type=str, default=None)

    # for hard negative sampling
    parser.add_argument("--hits_num", type=int, default=1000)

    parser.add_argument("--do_hn_gen", action="store_true")  # hard negative
    parser.add_argument("--hn_gen_num", type=int, default=1000)

    # model_param_list = add_model_hyperparameters(parser)  #  todo: enable
    model_param_list = []
    add_training_hyperparameters(parser)

    args = parser.parse_args()
    accelerator = setup_prerequisite(args)

    config, tokenizer = load_config_and_tokenizer(
        args, config_kwargs={
            # "problem_type": args.problem_type,
            # "num_labels": num_labels,
        })
    for param in model_param_list:
        setattr(config, param, getattr(args, param))

    encoder_class = AutoModel
    encoder = encoder_class.from_pretrained(args.model_name_or_path, config=config)
    embedding_dim = encoder.embedding_dim if hasattr(encoder, "embedding_dim") else len(tokenizer.get_vocab())
    model = encoder

    if args.do_train:
        model = DenseLearner(args, config, tokenizer, encoder, query_encoder=None)
        with accelerator.main_process_first():
            train_dataset = DatasetMarcoPassagesRanking(
                "train", args.data_dir, args.data_load_type, args, tokenizer, add_title=(not args.no_title))
        with accelerator.main_process_first():
            dev_dataset = DatasetRerank(
                "dev", args.data_dir, "memory", args, tokenizer, num_dev=args.num_dev, add_title=(not args.no_title))
        if args.negs_sources == "official":
            train_dataset.load_official_bm25_negatives(keep_num_neg=args.num_negs_per_system, )
        elif args.negs_sources.startswith("custom/"):
            negs_sources = args.negs_sources[7:]
            if negs_sources == "" or negs_sources == "all":
                all_neg_names = ["bm25-off", "co-stg1", "sp-stg1", "co-stg2", "sp-stg2"]
            else:
                all_neg_names = negs_sources.split(",")
            qid2varnegs = collections.defaultdict(collections.OrderedDict)  # aggregation
            for neg_name in all_neg_names:
                neg_filepath = os.path.join(args.custom_hn_dir, neg_name + ".pkl")
                if not file_exists(neg_filepath):
                    logger.warning(f"File Not Exist! {neg_filepath}")
                else:
                    for qid, neg_pids in load_pickle(neg_filepath).items():
                        qid2varnegs[int(qid)][neg_name] = neg_pids
            qid2negatives = dict()
            for qid, name2negs in qid2varnegs.items():
                agg_negs = []
                for negs in name2negs.values():
                    agg_negs.extend(negs[:args.num_negs_per_system])
                qid2negatives[qid] = agg_negs
            train_dataset.use_new_qid2negatives(qid2negatives, accelerator=None)
        else:
            # train_dataset.load_sbert_hard_negatives(
            #     ce_score_margin=args.ce_score_margin,
            #     num_negs_per_system=args.num_negs_per_system,
            #     negs_sources=args.negs_sources)
            raise NotImplementedError

        train(
            args, train_dataset, model, accelerator, tokenizer,
            eval_dataset=dev_dataset, eval_fn=evaluate_encoder_reranking)

    if args.do_eval or args.do_prediction or args.do_hn_gen:
        if args.do_train:
            encoder = encoder_class.from_pretrained(pretrained_model_name_or_path=args.output_dir, config=config)
        else:
            encoder = model
        encoder = accelerator.prepare(encoder)

        meta_best_str = ""
        if args.do_eval:
            with accelerator.main_process_first():
                if args.eval_reranking_source is None:
                    dev_dataset = DatasetRerank(
                        "dev", args.data_dir, "memory", args, tokenizer, num_dev=None, add_title=(not args.no_title))
                else:
                    dev_dataset = DatasetCustomRerank(
                        "dev", args.data_dir, "memory", args, tokenizer, num_dev=None, add_title=(not args.no_title),
                        filepath_dev_qid2top1000pids=args.eval_reranking_source,
                    )
            best_dev_result, best_dev_metric = evaluate_encoder_reranking(
                args, dev_dataset, encoder, accelerator, global_step=None,
                save_prediction=True, tokenizer=tokenizer, key_metric_name="MRR@10",
                similarity_metric=None, query_model=None,)
            if accelerator.is_local_main_process:
                # meta_best_str += f"best_test_result: {best_dev_result}, "
                meta_best_str += json.dumps(best_dev_metric) + os.linesep
        else:
            best_dev_result = None

        if args.do_prediction:
            best_pred_result, dev_pred_metric = evaluate_dense_retreival(
                args, None, encoder, accelerator, global_step=None, tb_writer=None, save_prediction=False,
                key_metric_name="MRR@10", delete_model=False, add_title=(not args.no_title), query_model=None,
                tokenizer=tokenizer, faiss_mode="gpu",
                hits=args.hits_num,
            )
            # meta_best_str += json.dumps(dev_pred_metric) + os.linesep

        if accelerator.is_local_main_process:
            with open(os.path.join(args.output_dir, "best_eval_results.txt"), "w") as fp:
                fp.write(f"{best_dev_result}, {meta_best_str}")

        if args.do_hn_gen:
            get_hard_negative_by_dense_retrieval(
                args, None, encoder, accelerator, global_step=None, tb_writer=None, save_prediction=False,
                key_metric_name="MRR@10", delete_model=False, add_title=(not args.no_title), query_model=None,
                tokenizer=tokenizer, faiss_mode="gpu",
                hits=args.hn_gen_num,
            )

if __name__ == '__main__':
    main()

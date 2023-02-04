import json
import tqdm

from peach.base import *

from peach.datasets.marco.dataset_marco_passages import DatasetMarcoPassagesRanking
from peach.datasets.marco.dataset_marco_eval import DatasetRerank, DatasetCustomRerank

from peach.enc_utils.eval_functions import evaluate_encoder_reranking
from peach.enc_utils.eval_sparse import evaluate_sparse_retreival
from peach.enc_utils.hn_gen_sparse import get_hard_negative_by_sparse_retrieval

from proj_sparse.modeling_splade_series import add_model_hyperparameters, DistilBertSpladeEnocder, \
    BertSpladeEnocder, ConSpladeEnocder, RobertaSpladeEnocder
from peach.enc_utils.enc_learners import LearnerMixin, FLOPS

import torch
import torch.nn as nn

from peach.enc_utils.sim_metric import Similarity
from peach.enc_utils.general import preproc_inputs
from peach.common import load_pickle

# from peach.nn_utils.general import gen_special_self_attn_mask, mask_out_cls_sep

def add_training_hyperparameters(parser):
    parser.add_argument("--lambda_d", type=float, default=0.0008)
    parser.add_argument("--lambda_q", type=float, default=0.0006)
    parser.add_argument("--apply_distill", action="store_true")


class SpladeLearner(LearnerMixin):
    def __init__(self, model_args, config, tokenizer, encoder, query_encoder=None):
        super().__init__(model_args, config, tokenizer, encoder, query_encoder, )

        self.sim_fct = Similarity(metric="dot")
        self.flops_loss = FLOPS()

    def forward(
            self,
            input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None,
            input_ids_query=None, attention_mask_query=None, token_type_ids_query=None, position_ids_query=None,
            distill_labels=None, training_progress=None,
            training_mode=None,
            **kwargs,
    ):

        if training_mode is None:
            return self.encoder(input_ids, attention_mask, ).contiguous()

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

        emb_doc, emb_query = doc_outputs["sentence_embedding"], query_outputs["sentence_embedding"]
        emb_dim = emb_doc.shape[-1]

        dict_for_loss = {}
        # calculate similarity and losses
        ga_emb_doc, ga_emb_query = self.gather_tensor(emb_doc), self.gather_tensor(emb_query)

        # if dist.is_initialized():
        dict_for_meta["flops_doc_loss_weight"] = self.model_args.lambda_d
        dict_for_meta["flops_query_loss_weight"] = self.model_args.lambda_q
        dict_for_meta["sparse_xentropy_loss_weight"] = float(self.my_world_size)

        loss_fct = nn.CrossEntropyLoss()
        target = torch.arange(ga_emb_query.shape[0], device=ga_emb_query.device, dtype=torch.long) * num_docs

        similarities = self.sim_fct(ga_emb_query, ga_emb_doc.view(-1, emb_dim))
        similarities.clamp_(max=10000)
        dict_for_loss["sparse_xentropy_loss"] = loss_fct(similarities, target)
        # sparse
        dict_for_loss["flops_doc_loss"] = self.flops_loss(emb_doc.view(-1, emb_dim))
        dict_for_loss["flops_query_loss"] = self.flops_loss(emb_query)

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

    parser.add_argument("--no_title", action="store_true")

    parser.add_argument("--encoder", type=str, default="distilbert", )

    parser.add_argument("--eval_reranking_source", type=str, default=None)

    parser.add_argument("--hits_num", type=int, default=1000)

    # for hard negative sampling
    parser.add_argument("--do_hn_gen", action="store_true")  # hard negative
    parser.add_argument("--hn_gen_num", type=int, default=1000)

    model_param_list = add_model_hyperparameters(parser)
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

    if args.encoder == "distilbert":
        encoder_class = DistilBertSpladeEnocder
    elif args.encoder == "bert":
        encoder_class = BertSpladeEnocder
    elif args.encoder == "condenser":
        encoder_class = ConSpladeEnocder
    elif args.encoder == "roberta":
        encoder_class = RobertaSpladeEnocder
    else:
        raise NotImplementedError(args.encoder)

    encoder = encoder_class.from_pretrained(args.model_name_or_path, config=config)
    embedding_dim = encoder.embedding_dim if hasattr(encoder, "embedding_dim") else len(tokenizer.get_vocab())
    model = encoder

    if args.do_train:
        model = SpladeLearner(args, config, tokenizer, encoder, query_encoder=None)
        with accelerator.main_process_first():
            train_dataset = DatasetMarcoPassagesRanking(
                "train", args.data_dir, args.data_load_type, args, tokenizer, add_title=(not args.no_title))
        with accelerator.main_process_first():
            dev_dataset = DatasetRerank(
                "dev", args.data_dir, "memory", args, tokenizer, num_dev=args.num_dev, add_title=(not args.no_title))
        if args.negs_sources == "official":
            train_dataset.load_official_bm25_negatives(keep_num_neg=args.num_negs_per_system, )
        elif args.negs_sources == "custom":
            assert args.negs_sources.startswith("custom/")
            negative_paths = args.negs_sources[7:].strip(";").split(";")
            qid2negatives = dict()
            for pkl_path in negative_paths:
                local_qid2negatives = load_pickle(pkl_path)
                for qid, lc_negs in local_qid2negatives.items():
                    qid = int(qid)
                    if qid not in qid2negatives:
                        qid2negatives[qid] = []
                    qid2negatives[qid].extend(lc_negs[:args.num_negs_per_system])
            train_dataset.use_new_qid2negatives(qid2negatives, accelerator=None)
        else:
            train_dataset.load_sbert_hard_negatives(
                ce_score_margin=args.ce_score_margin,
                num_negs_per_system=args.num_negs_per_system,
                negs_sources=args.negs_sources)
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

        # cannot do_prediction and do_hn_gen in one script!!!
        assert not (args.do_prediction and args.do_hn_gen), "cannot do_prediction and do_hn_gen in one script!!!"

        if args.do_prediction:
            best_pred_result, dev_pred_metric = evaluate_sparse_retreival(
                args, None, encoder, accelerator, global_step=None, tb_writer=None, save_prediction=False,
                key_metric_name="MRR@10", delete_model=False, add_title=(not args.no_title), query_model=None,
                # vocab_id2token=vocab_id2token,
                tokenizer=tokenizer, quantization_factor=100,
                hits=args.hits_num,
            )
            # meta_best_str += json.dumps(dev_pred_metric) + os.linesep

        if accelerator.is_local_main_process:
            with open(os.path.join(args.output_dir, "best_eval_results.txt"), "w") as fp:
                fp.write(f"{best_dev_result}, {meta_best_str}")

        if args.do_hn_gen:
            get_hard_negative_by_sparse_retrieval(
                args, None, encoder, accelerator, global_step=None, tb_writer=None, save_prediction=False,
                key_metric_name="MRR@10", delete_model=False, add_title=(not args.no_title), query_model=None,
                # vocab_id2token=vocab_id2token,
                tokenizer=tokenizer, quantization_factor=100,
                hits=args.hn_gen_num,
            )

if __name__ == '__main__':
    main()
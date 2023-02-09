# LED: Lexicon-Enlightened Dense Retriever for Large-Scale Retrieval
Source code of [LED: Lexicon-Enlightened Dense Retriever for Large-Scale Retrieval (WWW 2023)](https://arxiv.org/pdf/2208.13661.pdf)

# ![Workflow_LED](https://github.com/drogozhang/LED/blob/main/Workflow_LED.png)

# Env Setups

All these must be done in Conda Env

## Mandatory

```shell
conda create -n trans22 python=3.7
conda activate trans22
conda install pytorch=1.9.1 torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
# conda install pytorch=1.9.1 torchvision torchaudio -c pytorch
pip install transformers==4.11.3;

pip install gpustat ipython jupyter datasets accelerate sklearn tensorboard nltk
# pip install ipython jupyter datasets accelerate sklearn tensorboard nltk

conda install -c conda-forge faiss-gpu
```

### For LEX:

Please refer to [Anserini](https://github.com/castorini/anserini) to support LEX inference (for evaluation & self hard negative mining).

## Infrastructures Tested

- NVIDIA A100 80G

// you may need to re-search the hyperparameters when using multiple GPUs.

## Training

### Stage 1: Warm-up 

#### Dense Retriever (DEN (Warm-up)):

```shell
# first-stage dense retriever with best recipe we found
export DATA_DIR=[DATA_DIR]
CUDA_VISIBLE_DEVICES=7; python -m proj_dense.train_dense_retriever \
--do_train --do_eval --do_prediction --negs_sources official --num_negs_per_system 1000 \
--model_name_or_path Luyu/co-condenser-marco --overwrite_output_dir \
--warmup_proportion 0.01 --weight_decay 0.0 --max_grad_norm 1. --seed 42 \
--data_dir ${DATA_DIR} \
--output_dir ./output/dense/single-gpu-co-condenser-bs16nn7-lr1e5-ep3-qrylen32 \
--data_load_type memory --num_proc 6 --max_length 128 --eval_steps 10000 --logging_steps 100 \
--learning_rate 1e-5 --num_train_epochs 3 --eval_batch_size 32 --num_dev 1000 \
--train_batch_size 16 --gradient_accumulation_steps 1 --num_negatives 7 > logs/single-gpu-co-condenser-bs16nn7-lr1e5-ep3-qrylen32.log

# Do prediction
CUDA_VISIBLE_DEVICES=6,7; python -m torch.distributed.run --master_port 47784 --nproc_per_node=2 \
-m proj_dense.train_dense_retriever --do_prediction \
--model_name_or_path ./output/dense/single-gpu-co-condenser-bs16nn7-lr1e5-ep3-qrylen32 \
--overwrite_output_dir --data_dir ${DATA_DIR} \
--output_dir ./output/dense/single-gpu-co-condenser-bs16nn7-lr1e5-ep3-qrylen32 --eval_batch_size 64
```

#### Lexicon-aware Retriever (LEX (Warm-up)):

```shell
# first-stage lexicon-aware retriever with best recipe we found
export DATA_DIR=[DATA_DIR]
# official here indicates BM25 negatives.
python3 -m proj_sparse.train_splade_retriever \
--do_train --do_eval --encoder distilbert \
--model_name_or_path distilbert-base-uncased --overwrite_output_dir \
--warmup_proportion 0.01 --weight_decay 0.01 --max_grad_norm 1. --seed 42 \
--data_dir ${DATA_DIR} \
--output_dir ./output/sparse/splade-distilbert-bs48nn5-NSoff1000-lr3e5-ep3-baseline \
--data_load_type memory --num_proc 2 --max_length 128 --eval_steps 10000 --logging_steps 100 \
--learning_rate 3e-5 --num_train_epochs 3 --eval_batch_size 48 --num_dev 1000 \
--train_batch_size 48 --gradient_accumulation_steps 1 --num_negatives 5 \
--num_negs_per_system 1000 --negs_sources official > logs/splade-distilbert-bs48nn5-NSoff1000-lr3e5-ep3-baseline.log

# --do_prediction separately if using DDP
python3 -m proj_sparse.train_splade_retriever \
--do_prediction --encoder distilbert \
--model_name_or_path ./output/sparse/splade-distilbert-bs48nn5-NSoff1000-lr3e5-ep3-baseline --overwrite_output_dir \
--data_dir ${DATA_DIR} \
--output_dir ./output/sparse/splade-distilbert-bs48nn5-NSoff1000-lr3e5-ep3-baseline \
--data_load_type memory --num_proc 2 --max_length 128 --eval_batch_size 48

```



### Stage 2: Continue Training

#### Dense Retriever (DEN (Continue))

##### Step 1: Generate hard negatives

```shell
export DATA_DIR=[DATA_DIR]

python -m torch.distributed.run --master_port 47778 --nproc_per_node=4 \
 -m proj_dense.generate_dense_hard_negs \
--num_negatives 200 --overwrite_save_path \
--model_name_or_path ./output/dense/single-gpu-co-condenser-bs16nn7-lr1e5-ep3-qrylen32-stage2 \
--data_dir ${DATA_DIR} \
--output_dir ./output/dense/2stage/co-condenser-bs16nn7-lr1e5-ep3-qrylen32-stage2 \
--data_load_type memory --num_proc 6 --eval_batch_size 256
```

##### Step 2: Continue Training

```shell
# second-stage dense retriever with best recipe we found
export DATA_DIR=[DATA_DIR]

python -m proj_dense.train_dense_retriever_continue \
--static_hard_negatives_path ./output/dense/2stage/co-condenser-bs16nn7-lr1e5-ep3-qrylen32-stage2 \
--do_train --do_eval --do_prediction --num_negs_per_system 200 \
--model_name_or_path ./output/dense/single-gpu-co-condenser-bs16nn7-lr1e5-ep3-qrylen32-stage2 --overwrite_output_dir \
--warmup_proportion 0.01 --weight_decay 0.0 --max_grad_norm 1. --seed 42 \
--data_dir ${DATA_DIR} \
--output_dir ./output/dense/2stage/single-gpu-co-condenser-bs16nn7-lr5e6-ep3-qrylen32-stage2 \
--data_load_type memory --num_proc 6 --max_length 128 --eval_steps 10000 --logging_steps 100 \
--learning_rate 5e-6 --num_train_epochs 3 --eval_batch_size 32 --num_dev 1000 \
--train_batch_size 16 --gradient_accumulation_steps 1 --num_negatives 7
```

#### Lexicon-Aware Retriever (LEX (Continue) and Lexicon teacher)

##### Step 1: Generate hard negatives

```shell
export DATA_DIR=[DATA_DIR]

python -m proj_sparse.generate_sparse_hard_negs \
--data_dir ${DATA_DIR} --eval_batch_size 64 --output_dir ./output/sparse/splade-distilbert-bs48nn5-NSoff1000-lr3e5-ep3-baseline \
--model_name_or_path ./output/sparse/splade-distilbert-bs48nn5-NSoff1000-lr3e5-ep3-baseline \
--num_negatives 200 --encoder distilbert --data_load_type memory --overwrite_save_path

# (batchlize train query search because anserini works very slow after 200k search. Do this with several running, so possibly code has bugs. But current file is fine.)
```

##### Step 2: Train Stage 2 Model with Static Hard Negs

```shell
# second-stage lexicon-aware retriever with best recipe we found
export DATA_DIR=[DATA_DIR]

python -m proj_sparse.train_splade_retriever_continue \
--static_hard_negatives_path ./output/sparse/splade-distilbert-bs48nn5-NSoff1000-lr3e5-ep3-baseline \
--do_train --do_eval --do_prediction --encoder distilbert \
--model_name_or_path ./output/sparse/splade-distilbert-bs48nn5-NSoff1000-lr3e5-ep3-baseline --overwrite_output_dir \
--warmup_proportion 0.01 --weight_decay 0.01 --max_grad_norm 1. --seed 42 \
--data_dir ${DATA_DIR} \
--output_dir ./output/sparse/2stage/splade-distilbert-bs48nn5-NSoff1000-lr2e5-ep3-baseline \
--data_load_type memory --num_proc 2 --max_length 128 --eval_steps 10000 --logging_steps 100 \
--learning_rate 2e-5 --num_train_epochs 3 --eval_batch_size 48 --num_dev 1000 \
--train_batch_size 48 --gradient_accumulation_steps 1 --num_negatives 5 \
--num_negs_per_system 200
```

### LED Training

```shell
# LED retriever with best recipe we found  # in practice, we found NN=30 is even better.
export DATA_DIR = [DATA_DIR]
export CUSTOM_HN_DIR = [CUSTOM_HN_DIR]
export NN=32
export DSTW=1.0
export DSTM=rank
export NEGS=co-stg1,sp-stg1,sp-stg2

python -m proj_dense.train_LED_retriever \
--tch_model_path ./output/sp-stg2 \
--tch_encoder distilbert \
--with_pos \
--apply_dst \
--tch_no_drop \
--dst_method ${DSTM} \
--dst_loss_weight ${DSTW} \
--negs_sources custom/${NEGS} \
--do_train --do_eval --do_prediction --num_negs_per_system 200 \
--model_name_or_path ./output/co-stg1 \
--overwrite_output_dir \
--warmup_proportion 0.01 \
--weight_decay 0.0 \
--max_grad_norm 1. \
--seed 42 \
--data_dir ${DATA_DIR} \
--custom_hn_dir ${CUSTOM_HN_DIR} \
--output_dir ./output/entangle/stg1-co-bs16nn${NN}-lr5e6-ep3-sp-stg2-dst-${DSTM}-with-pos-${DSTW}-${NEGS}-tch-no-drop \
--data_load_type memory --num_proc 6 --max_length 128 --eval_steps 10000 --logging_steps 100 \
--learning_rate 5e-6 --num_train_epochs 3 --eval_batch_size 32 --num_dev 1000 \
--train_batch_size 16 --gradient_accumulation_steps 1 --num_negatives ${NN}
```

### LED (w/ RT) Training

```shell
export DATA_DIR = [DATA_DIR]
export CUSTOM_HN_DIR = [CUSTOM_HN_DIR]
export TCH_PATH=[TCH_PATH]
export DSTM=rank
export DSTW=1.2
export TEMP=3.0
export XDSTW=1.5
export NN=32
export NEGS=co-stg1,sp-stg1,sp-stg2;
# sp-stg2 is the LEX teacher, co-stg1 is the DEN after warming up, both obtained from previous steps.

python -m proj_dense.train_LED_w_RT \
--sp_tch_model_path ./output/sp-stg2 \
--sp_tch_encoder distilbert \
--apply_dst \
--with_ce_loss \
--dst_method ${DSTM} \
--xe_tch_temp ${TEMP} \
--tch_no_drop \
--dst_loss_weight ${DSTW} \
--xe_tch_model_path  \
--xe_tch_dst_loss_weight ${XDSTW} \
--negs_sources custom/${NEGS} \
--num_negs_per_system 200 \
--do_train --do_eval --do_prediction \
--model_name_or_path ./output/co-stg1 \
--overwrite_output_dir \
--warmup_proportion 0.01 \
--weight_decay 0.0 \
--max_grad_norm 1. \
--seed 42 \
--data_dir ${CORPUS_DIR} \
--custom_hn_dir ${CUSTOM_HN_DIR} \
--output_dir ./output/entangle/stg1-co-bs16nn${NN}-lr5e6-ep3-sp-stg2-dst-${DSTM}-${DSTW}-${NEGS}-xencoder-kl-${XDSTW}-tempv2-${TEMP}-tch-no-drop \
--data_load_type memory --num_proc 6 --max_length 128 --eval_steps 10000 --logging_steps 100 \
--learning_rate 5e-6 --num_train_epochs 3 --eval_batch_size 32 --num_dev 1000 \
--train_batch_size 16 --gradient_accumulation_steps 1 --num_negatives ${NN}
```

## Citation

If you find our code helpful, please cite our paper: (the url is not available yet)


```
@inproceedings{Zhang2023LED,
  title={LED: Lexicon-Enlightened Dense Retriever for Large-Scale Retrieval},
  author={Kai Zhang, Chongyang Tao, Tao Shen, Can Xu, Xiubo Geng, Binxing Jiao, Daxin Jiang},
  booktitle={Proceedings of WWW 2023},
  url      ={https://doi.org/10.1145/3543507.3583294},
  doi      ={10.1145/3543507.3583294},
  year={2023}
}
```

## Question

If you find any questions, please feel free to contact Kai Zhang `drogozhang@gmail.com`.

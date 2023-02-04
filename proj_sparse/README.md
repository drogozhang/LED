### Special Notes:
1. in stage 1 training. do_prediciton solely and do_prediciton while training are different.
2. in both stages, performance of training in a single GPU is always better than that of DDP.

#### Best Stage 1 Training Script
python3 -m proj_sparse.train_splade_retriever \
--do_train --do_eval --encoder distilbert \
--model_name_or_path distilbert-base-uncased --overwrite_output_dir \
--warmup_proportion 0.01 --weight_decay 0.01 --max_grad_norm 1. --seed 42 \
--data_dir /relevance2-nfs/shentao/text_corpus/doc_pretrain_corpus \
--output_dir ./output/sparse/splade-distilbert-bs48nn5-NSoff1000-lr3e5-ep3-baseline \
--data_load_type memory --num_proc 2 --max_length 128 --eval_steps 10000 --logging_steps 100 \
--learning_rate 3e-5 --num_train_epochs 3 --eval_batch_size 48 --num_dev 1000 \
--train_batch_size 48 --gradient_accumulation_steps 1 --num_negatives 5 \
--num_negs_per_system 1000 --negs_sources official

##### do_prediction
python3 -m proj_sparse.train_splade_retriever \
--do_prediction --encoder distilbert \
--model_name_or_path ./output/sparse/splade-distilbert-bs48nn5-NSoff1000-lr3e5-ep3-baseline --overwrite_output_dir \
--data_dir /relevance2-nfs/shentao/text_corpus/doc_pretrain_corpus \
--output_dir ./output/sparse/splade-distilbert-bs48nn5-NSoff1000-lr3e5-ep3-baseline \
--data_load_type memory --num_proc 2 --max_length 128 --eval_batch_size 48

#### Generate Static Hard negative Script
python -m proj_sparse.generate_sparse_hard_negs --data_dir /relevance2-nfs/shentao/text_corpus/doc_pretrain_corpus --eval_batch_size 64 --output_dir ./output/sparse/splade-distilbert-bs48nn5-NSoff1000-lr3e5-ep3-baseline \
--model_name_or_path ./output/sparse/splade-distilbert-bs48nn5-NSoff1000-lr3e5-ep3-baseline \
--num_negatives 200 --encoder distilbert --data_load_type memory --overwrite_save_path

(batchlize search because anserini works very slow after 200k train query search. )

#### Train STAR:

python -m proj_sparse.train_splade_retriever_2stage_static_neg \
--static_hard_negatives_path ./output/sparse/splade-distilbert-bs48nn5-NSoff1000-lr3e5-ep3-baseline \
--do_train --do_eval --do_prediction --encoder distilbert \
--model_name_or_path ./output/sparse/splade-distilbert-bs48nn5-NSoff1000-lr3e5-ep3-baseline --overwrite_output_dir \
--warmup_proportion 0.01 --weight_decay 0.01 --max_grad_norm 1. --seed 42 \
--data_dir /relevance2-nfs/shentao/text_corpus/doc_pretrain_corpus \
--output_dir ./output/sparse/2stage/splade-distilbert-bs48nn5-NSoff1000-lr2e5-ep3-baseline \
--data_load_type memory --num_proc 2 --max_length 128 --eval_steps 10000 --logging_steps 100 \
--learning_rate 2e-5 --num_train_epochs 3 --eval_batch_size 48 --num_dev 1000 \
--train_batch_size 48 --gradient_accumulation_steps 1 --num_negatives 5

------
##### Notes:
1. Not mixing BM25 official negatives is better in stage 2 (better 0.3)
2. learning rate in stage 2 is smaller than that in stage 1, approaximately half. (better 0.5)
3. Other hyperparameters don't influence a lot.


#### Causions:
If use DDP training, DON'T do_prediciton and do_train at the same time. Run 'do_train and do_eval' and 'do_prediction' in two separate scripts!!
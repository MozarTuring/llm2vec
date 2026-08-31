JWM_PYTHON="3.10"
JWM_SERVER_NAME=greatrawr
JWM_build_flashattn=
JWM_NOTEBOOK=
CUDA_VISIBLE_DEVICES=1

# JWM_RUN_COMMAND="python experiments/hard_negatives.py"

# JWM_RUN_COMMAND="python experiments/reranker.py msmarco_hard_negatives.json --output reranked_hard_negatives.json --top_k 8"

JWM_RUN_COMMAND="python experiments/run_layerwise_finetune.py train_configs/layerwise/MetaLlama3.1-mntp-layerwise.json"

# JWM_RUN_COMMAND="python experiments/diag_sae.py"

# JWM_RUN_COMMAND="python experiments/mteb_eval_layerwise.py \
#   --config output/layerwise/Meta-Llama-3.1-8B-msmarco-mntp-L0/MetaLlama3.1-mntp-layerwise.json \
#   --trained_checkpoint_path output/layerwise/Meta-Llama-3.1-8B-msmarco-mntp-L0/checkpoint-3929 \
#   --query_top_k 40 \
#   --doc_top_k 400 \
#   --output_dir results \
#   --cache_dir embedding_cache \
#   --max_length 1024 \
#   --task_name SciFact"
#




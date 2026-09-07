set -e 
export JWM_PYTHON="3.10"
export JWM_SERVER_NAME=greatrawr
export JWM_build_flashattn=
export JWM_NOTEBOOK=
export CUDA_VISIBLE_DEVICES=1
export JWM_RUN_COMMAND="python experiments/hard_negatives.py --top-k 1000 --num-top 50 --num-random 50 --output msmarco_hard_negatives_v2.json --query_batch_size 1024"

set -e 
export JWM_SERVER_NAME=greatrawr
export CUDA_VISIBLE_DEVICES=1
export JWM_RUN_COMMAND="python experiments/run_layerwise_finetune.py train_configs/layerwise/MetaLlama3.1-mntp-layerwise.json"
export JWM_PYTHON="3.10"
export JWM_NOTEBOOK=
export JWM_build_flashattn=

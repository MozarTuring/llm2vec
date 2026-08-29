set -e 
export JWM_SERVER_NAME=berzeliusampere
export JWM_GPU_NUM=1
export JWM_RUN_TIME="1-00:00:00"
export JWM_SLURM_FILE=slurm.sh
export JWM_NOTEBOOK=
export JWM_NODES_NUM=1
export JWM_RUN_COMMAND="python experiments/run_layerwise_finetune.py train_configs/layerwise/MetaLlama3.1-mntp-layerwise.json"
export JWM_build_flashattn=

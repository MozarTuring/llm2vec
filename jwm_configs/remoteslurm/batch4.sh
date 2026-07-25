JWM_SERVER_NAME=berzeliusampere
JWM_GPU_NUM=1
JWM_NODES_NUM=1
JWM_RUN_TIME="1-00:00:00"
JWM_SLURM_FILE=slurm.sh
JWM_build_flashattn=
JWM_NOTEBOOK=1


JWM_SLURM_RUN_COMMAND="python experiments/run_layerwise_finetune.py"

JWM_SLURM_RUN_ARGS="train_configs/layerwise/MetaLlama3.1-mntp-layerwise.json"

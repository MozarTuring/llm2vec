JWM_SERVER_NAME=berzeliusampere
JWM_GPU_NUM=1
JWM_NODES_NUM=1
JWM_RUN_TIME="0-10:00:00"
JWM_SLURM_FILE=slurm.sh
JWM_build_flashattn=


JWM_SLURM_RUN_COMMAND="python experiments/run_word_task.py"

# JWM_SLURM_RUN_ARGS="train_configs/word-task/ShearedLlama-bi-mntp.json"
# JWM_SLURM_RUN_ARGS="train_configs/word-task/ShearedLlama-bi.json"
# JWM_SLURM_RUN_ARGS="train_configs/word-task/Llama2-bi-mntp.json"
JWM_SLURM_RUN_ARGS="train_configs/word-task/MetaLlama3.1-bi-mntp.json"


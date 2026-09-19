PKQ_SERVER_NAME=berzeliusampere
PKQ_GPU_NUM=1
PKQ_NODES_NUM=1
PKQ_RUN_TIME="0-10:00:00"
PKQ_SLURM_FILE=slurm.sh
PKQ_build_flashattn=


PKQ_SLURM_RUN_COMMAND="python experiments/run_word_task.py"

# PKQ_SLURM_RUN_ARGS="train_configs/word-task/ShearedLlama-bi-mntp.json"
# PKQ_SLURM_RUN_ARGS="train_configs/word-task/ShearedLlama-bi.json"
# PKQ_SLURM_RUN_ARGS="train_configs/word-task/Llama2-bi-mntp.json"
# PKQ_SLURM_RUN_ARGS="train_configs/word-task/MetaLlama3.1-bi-mntp.json"
PKQ_SLURM_RUN_ARGS="train_configs/word-task/MetaLlama3_1-bi.json"


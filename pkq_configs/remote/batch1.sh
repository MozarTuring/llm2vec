PKQ_SERVER_NAME=berzeliusampere
PKQ_GPU_NUM=2
PKQ_NODES_NUM=1
PKQ_RUN_TIME="3-00:00:00"
PKQ_SLURM_FILE=slurm.sh
PKQ_build_flashattn=


PKQ_SLURM_RUN_COMMAND="python experiments/run_mntp.py"

#PKQ_SLURM_RUN_ARGS="train_configs/mntp/Sheared-Llama.json"
#PKQ_SLURM_RUN_ARGS="train_configs/mntp/Mistral.json"
#PKQ_SLURM_RUN_ARGS="train_configs/mntp/MetaLlama3.json"
PKQ_SLURM_RUN_ARGS="train_configs/mntp/MetaLlama3.1-msmarco.json"

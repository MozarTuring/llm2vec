
set -e
# change the following vars based on your preference, and then make sure this repo is cloned to /nobackup/proj/disk/naiss2026-3-658/personal/jinma63/project_remote_jwm/llm2vec_jingwei
export RUN_DIR_HOME=/nobackup/proj/disk/naiss2026-3-658/personal/jinma63
export RUN_PROJ=llm2vec_jingwei
export JWM_DATA_DIR=/nobackup/proj/disk/naiss2026-3-658/personal/jinma63/project_remote_jwm/remote_data/llm2vec

cd ${RUN_DIR_HOME}/project_remote_jwm/${RUN_PROJ}
export JWM_PYTHON="3.10"
export JWM_SERVER_NAME=arrhenius
export JWM_GPU_NUM=1
export JWM_NODES_NUM=1
export JWM_RUN_TIME="1-00:00:00"
export JWM_SLURM_FILE=slurm.sh
export JWM_build_flashattn=
export JWM_NOTEBOOK=
export MEM_PER_TASK="$((30 * JWM_GPU_NUM))G"
export CPUS_PER_TASK=$((8 * JWM_GPU_NUM))
export CUDA_VISIBLE_DEVICES=1
export JWM_TASK_NAMES="ArguAna CQADupstackGamingRetrieval CQADupstackUnixRetrieval ClimateFEVERHardNegatives FEVERHardNegatives FiQA2018 HotpotQAHardNegatives SCIDOCS TRECCOVID Touche2020Retrieval.v3"
export JWM_RUN_COMMAND="python experiments/mteb_eval_layerwise.py   --trained_checkpoint_path ${JWM_DATA_DIR}/output/layerwise/Meta-Llama-3.1-8B-msmarco-mntp-L0/checkpoint-3929   --query_top_k 40   --doc_top_k 400   --output_dir results   --max_length 1024   --task_name ${JWM_TASK_NAMES}"


set -e
# set following vars based on your setting, and then make sure this repo is cloned to /proj/berzelius-aiics-real/users/x_jinma/project_remote_pkq/llm2vec_pikaq
export RUN_DIR_HOME=
export RUN_PROJ=

cd ${RUN_DIR_HOME}/project_remote_pkq/${RUN_PROJ}
export PKQ_PYTHON="3.10"
export PKQ_SERVER_NAME=berzeliusampere
export PKQ_GPU_NUM=4
export PKQ_NODES_NUM=1
export PKQ_RUN_TIME="1-00:00:00"
export PKQ_SLURM_FILE=slurm.sh
export PKQ_build_flashattn=
export PKQ_NOTEBOOK=
export PKQ_INTERACTIVE=
export MEM_PER_TASK="$((80 * PKQ_GPU_NUM))G"
export CPUS_PER_TASK=$((8 * PKQ_GPU_NUM))
export CUDA_VISIBLE_DEVICES=1
export PKQ_TASK_NAMES="ArguAna"
export PKQ_RUN_COMMAND="python experiments/mteb_eval_layerwise.py --trained_checkpoint_path ${PKQ_DATA_DIR}/backup/arrhenius/20260926_074205/output/layerwise/Meta-Llama-3.1-8B-msmarco-mntp-L26/checkpoint-3930 --query_top_k 40 --doc_top_k 400 --output_dir results --max_length 1024 --task_name ${PKQ_TASK_NAMES}"

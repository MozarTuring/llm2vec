
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
export PKQ_RUN_COMMAND="python experiments/fuse_eval.py --dump_dir ${PKQ_DATA_DIR}/backup/berzeliusampere/20261009_103719/sparse_dumps ${PKQ_DATA_DIR}/backup/berzeliusampere/20261009_105202/sparse_dumps --layers 0 26 --output_dir results_fusion"

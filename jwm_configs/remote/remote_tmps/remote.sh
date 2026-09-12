
set -e
# change the following vars based on your preference
export RUN_DIR_HOME=/home/jinma
export RUN_PROJ=llm2vec_jingwei
export JWM_DATA_DIR=/home/jinma/project_remote_jwm/remote_data/llm2vec

cd ${RUN_DIR_HOME}/project_remote_jwm/${RUN_PROJ}
export JWM_PYTHON="3.10"
export JWM_SERVER_NAME=greatrawr
export JWM_GPU_NUM=1
export JWM_NODES_NUM=1
export JWM_RUN_TIME="1-00:00:00"
export JWM_SLURM_FILE=slurm.sh
export JWM_build_flashattn=
export JWM_NOTEBOOK=
export CUDA_VISIBLE_DEVICES=1
export JWM_RUN_COMMAND="python experiments/run_layerwise_finetune.py     train_configs/layerwise/MetaLlama3.1-mntp-layerwise.json     --hard_negatives_file ${JWM_DATA_DIR}/reranked_parts/     --output_dir ${JWM_DATA_DIR}/output/layerwise/"

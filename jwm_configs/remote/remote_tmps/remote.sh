set -e 
export JWM_PYTHON="3.10"
export JWM_SERVER_NAME=greatrawr
export JWM_MODE=remotenone
export JWM_GPU_NUM=1
export JWM_NODES_NUM=1
export JWM_RUN_TIME="1-00:00:00"
export JWM_SLURM_FILE=slurm.sh
export JWM_build_flashattn=
export JWM_NOTEBOOK=
export JWM_MODULES="Miniforge3 buildenv-gcccuda/12.4.1-gcc13.3.0"
export JWM_SLURM_NODES="--nodelist=node[061-064,065,066-093]"
export CUDA_VISIBLE_DEVICES=1
export JWM_RUN_COMMAND="python experiments/reranker.py split ${JWM_DATA_DIR}/msmarco_hard_negatives_v2.json --num-parts 100 --output-dir ${JWM_DATA_DIR}/msmarco_hard_negatives_v2_parts/"

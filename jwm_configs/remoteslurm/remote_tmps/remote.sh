set -e 
export JWM_SERVER_NAME=berzeliusampere
export JWM_GPU_NUM=4
export JWM_NODES_NUM=1
export JWM_RUN_TIME="1-00:00:00"
export JWM_SLURM_FILE=slurm.sh
export JWM_build_flashattn=
export JWM_NOTEBOOK=
export JWM_SLURM_NODES="--nodelist=node[061-064,065,066-093]"
export JWM_RUN_COMMAND="python experiments/reranker.py ${JWM_DATA_DIR}/msmarco_hard_negatives_v2.json --output ${JWM_DATA_DIR}/reranked_hard_negatives_v2.json --batch_size 3500"

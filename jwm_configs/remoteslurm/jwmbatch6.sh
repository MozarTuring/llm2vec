JWM_SERVER_NAME=berzeliusampere
JWM_GPU_NUM=4
JWM_NODES_NUM=1
JWM_RUN_TIME="3-00:00:00"
JWM_SLURM_FILE=slurm.sh
JWM_build_flashattn=
JWM_NOTEBOOK=
JWM_SLURM_NODES="--nodelist=node[061-064,065,066-093]"

# JWM_RUN_COMMAND="python experiments/hard_negatives.py --top-k 1000 --num-top 50 --num-random 50 --output ${JWM_DATA_DIR}/msmarco_hard_negatives_v2.json --query-batch-size 8192"

JWM_RUN_COMMAND="python experiments/reranker.py split ${JWM_DATA_DIR}/msmarco_hard_negatives_v2.json --num-parts 10 --output-dir ${JWM_DATA_DIR}/msmarco_hard_negatives_v2_parts/"

# JWM_RUN_COMMAND="python experiments/reranker.py ${JWM_DATA_DIR}/msmarco_hard_negatives_v2.json --output ${JWM_DATA_DIR}/reranked_hard_negatives_v2.json --queries-per-batch 5"

# JWM_RUN_COMMAND="python experiments/run_layerwise_finetune.py train_configs/layerwise/MetaLlama3.1-mntp-layerwise.json"

# JWM_RUN_COMMAND="python experiments/diag_sae.py"


# BEIR 13 tasks (Table 5 in paper)
# JWM_TASK_NAMES="SciFact ArguAna ClimateFEVER DBPedia FEVER FiQA2018 HotpotQA NFCorpus NQ QuoraRetrieval SCIDOCS TRECCOVID Touche2020"

# MTEB(Eng, v2) retrieval tasks (Table 6 in paper, used in Figure 2)
# JWM_TASK_NAMES="ArguAna CQADupstackGamingRetrieval CQADupstackUnixRetrieval ClimateFEVERHardNegatives FEVERHardNegatives FiQA2018 HotpotQAHardNegatives SCIDOCS TRECCOVID Touche2020Retrieval.v3"
#
# JWM_RUN_COMMAND="python experiments/mteb_eval_layerwise.py \
#   --config output/layerwise/Meta-Llama-3.1-8B-msmarco-mntp-L0/MetaLlama3.1-mntp-layerwise.json \
#   --trained_checkpoint_path output/layerwise/Meta-Llama-3.1-8B-msmarco-mntp-L0/checkpoint-3929 \
#   --query_top_k 40 \
#   --doc_top_k 400 \
#   --output_dir results \
#   --max_length 1024 \
#   --task_name ${JWM_TASK_NAMES}"


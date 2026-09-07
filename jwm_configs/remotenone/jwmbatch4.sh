JWM_PYTHON="3.10"
JWM_SERVER_NAME=greatrawr
JWM_build_flashattn=
JWM_NOTEBOOK=
CUDA_VISIBLE_DEVICES=1

JWM_RUN_COMMAND="python experiments/hard_negatives.py --top-k 1000 --num-top 50 --num-random 50 --output msmarco_hard_negatives_v2.json --query_batch_size 1024"

# JWM_RUN_COMMAND="python experiments/reranker.py msmarco_hard_negatives.json --output reranked_hard_negatives.json --top_k 8"

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

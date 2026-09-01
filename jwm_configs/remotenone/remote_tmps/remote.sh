set -e 
export JWM_PYTHON="3.10"
export JWM_NOTEBOOK=
export JWM_build_flashattn=
export CUDA_VISIBLE_DEVICES=1
export JWM_TASK_NAMES="SciFact ArguAna ClimateFEVER DBPedia FEVER FiQA2018 HotpotQA NFCorpus NQ QuoraRetrieval SCIDOCS TRECCOVID Touche2020" #
export JWM_SERVER_NAME=greatrawr
export JWM_RUN_COMMAND="python experiments/mteb_eval_layerwise.py   --config output/layerwise/Meta-Llama-3.1-8B-msmarco-mntp-L0/MetaLlama3.1-mntp-layerwise.json   --trained_checkpoint_path output/layerwise/Meta-Llama-3.1-8B-msmarco-mntp-L0/checkpoint-3929   --query_top_k 40   --doc_top_k 400   --output_dir results   --cache_dir embedding_cache   --max_length 1024   --task_name ${JWM_TASK_NAMES}"

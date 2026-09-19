PKQ_PYTHON="3.10"
# PKQ_SERVER_NAME=berzeliusampere
PKQ_SERVER_NAME=arrhenius
# PKQ_SERVER_NAME=greatrawr
PKQ_GPU_NUM=2
PKQ_NODES_NUM=1
PKQ_RUN_TIME="1-00:00:00"
PKQ_SLURM_FILE=slurm.sh
PKQ_build_flashattn=
PKQ_NOTEBOOK=
PKQ_INTERACTIVE=1

MEM_PER_TASK="$((30 * PKQ_GPU_NUM))G"
CPUS_PER_TASK=$((8 * PKQ_GPU_NUM))

CUDA_VISIBLE_DEVICES=1

# PKQ_RUN_COMMAND="python experiments/hard_negatives.py --top-k 1000 --num-top 50 --num-random 50 --output ${PKQ_DATA_DIR}/msmarco_hard_negatives_v2.json --query-batch-size 8192"

# PKQ_RUN_COMMAND="python experiments/reranker.py split ${PKQ_DATA_DIR}/msmarco_hard_negatives_v2.json --num-parts 100 --output-dir ${PKQ_DATA_DIR}/msmarco_hard_negatives_v2_parts/"

# PKQ_RUN_COMMAND="python experiments/reranker.py rerank ${PKQ_DATA_DIR}/msmarco_hard_negatives_v2_parts --queries-per-batch 2 --output ${PKQ_DATA_DIR}/reranker_parts"

# PKQ_RUN_COMMAND="torchrun --nproc_per_node=${PKQ_GPU_NUM} experiments/run_layerwise_finetune.py \
#     --config train_configs/layerwise/MetaLlama3.1-mntp-layerwise.json \
#     --hard_negatives_file ${PKQ_DATA_DIR}/reranker_parts/"

# BEIR 13 tasks (Table 5 in paper)
# PKQ_TASK_NAMES="SciFact ArguAna ClimateFEVER DBPedia FEVER FiQA2018 HotpotQA NFCorpus NQ QuoraRetrieval SCIDOCS TRECCOVID Touche2020"

# MTEB(Eng, v2) retrieval tasks (Table 6 in paper, used in Figure 2)
PKQ_TASK_NAMES="ArguAna CQADupstackGamingRetrieval CQADupstackUnixRetrieval ClimateFEVERHardNegatives FEVERHardNegatives FiQA2018 HotpotQAHardNegatives SCIDOCS TRECCOVID Touche2020Retrieval.v3"

PKQ_RUN_COMMAND="python experiments/mteb_eval_layerwise.py \
  --trained_checkpoint_path ${PKQ_DATA_DIR}/output/layerwise/Meta-Llama-3.1-8B-msmarco-mntp-L0/checkpoint-3930 \
  --query_top_k 40 \
  --doc_top_k 400 \
  --output_dir results \
  --max_length 1024 \
  --task_name ${PKQ_TASK_NAMES}"


# PKQ_RUN_COMMAND="python experiments/check_postprocessing.py $PKQ_DATA_DIR/Llama3_1-8B-Base-L0R-8x"

# PKQ_RUN_COMMAND="python experiments/diag_sae.py"

# PKQ_RUN_COMMAND="python experiments/check_sae_threshold.py \
#     --sae_weights_path ${PKQ_DATA_DIR}/Llama3_1-8B-Base-L0R-8x/checkpoints/final.safetensors"



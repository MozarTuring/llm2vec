PKQ_PYTHON="3.10"
PKQ_SERVER_NAME=berzeliushopper
PKQ_SERVER_NAME=arrhenius
PKQ_SERVER_NAME=berzeliusampere
# PKQ_SERVER_NAME=greatrawr
PKQ_GPU_NUM=4
PKQ_NODES_NUM=1
PKQ_RUN_TIME="1-00:00:00"
PKQ_SLURM_FILE=slurm.sh
PKQ_build_flashattn=
PKQ_NOTEBOOK=
PKQ_INTERACTIVE=

MEM_PER_TASK="$((80 * PKQ_GPU_NUM))G"
CPUS_PER_TASK=$((8 * PKQ_GPU_NUM))

CUDA_VISIBLE_DEVICES=1

# PKQ_RUN_COMMAND="torchrun --nproc_per_node=${PKQ_GPU_NUM} experiments/run_mntp.py train_configs/mntp/MetaLlama3.1-msmarco.json"


# PKQ_RUN_COMMAND="python experiments/hard_negatives.py --top-k 1000 --num-top 50 --num-random 50 --output ${PKQ_DATA_DIR}/msmarco_hard_negatives_v2.json --query-batch-size 8192"

# PKQ_RUN_COMMAND="python experiments/reranker.py split ${PKQ_DATA_DIR}/msmarco_hard_negatives_v2.json --num-parts 100 --output-dir ${PKQ_DATA_DIR}/msmarco_hard_negatives_v2_parts/"

# PKQ_RUN_COMMAND="python experiments/reranker.py rerank ${PKQ_DATA_DIR}/msmarco_hard_negatives_v2_parts --queries-per-batch 2 --output ${PKQ_DATA_DIR}/reranker_parts"

# PKQ_RUN_COMMAND="torchrun --nproc_per_node=${PKQ_GPU_NUM} experiments/run_layerwise_finetune.py \
#     --config train_configs/layerwise/MetaLlama3.1-mntp-layerwise.json \
#     --hard_negatives_file ${PKQ_DATA_DIR}/reranker_parts/"


# BEIR 13 tasks (Table 5 in paper)
# PKQ_TASK_NAMES="SciFact ArguAna ClimateFEVER DBPedia FEVER FiQA2018 HotpotQA NFCorpus NQ QuoraRetrieval SCIDOCS TRECCOVID Touche2020"

# PKQ_TASK_NAMES="FEVERHardNegatives HotpotQAHardNegatives Touche2020Retrieval.v3 ClimateFEVERHardNegatives"

# PKQ_TASK_NAMES="ArguAna CQADupstackGamingRetrieval CQADupstackUnixRetrieval ClimateFEVERHardNegatives FEVERHardNegatives FiQA2018 HotpotQAHardNegatives SCIDOCS TRECCOVID Touche2020Retrieval.v3"
# PKQ_RUN_COMMAND="python experiments/mteb_eval_layerwise.py \
# --trained_checkpoint_path ${PKQ_DATA_DIR}/backup/arrhenius/20260926_074205/output/layerwise/Meta-Llama-3.1-8B-msmarco-mntp-L26/checkpoint-3930 \
# --query_top_k 40 \
# --doc_top_k 400 \
# --output_dir results \
# --max_length 1024 \
# --task_name ${PKQ_TASK_NAMES}"



# PKQ_RUN_COMMAND="python experiments/dump_sparse.py --trained_checkpoint_path ${PKQ_DATA_DIR}/backup/20261010_125818/output/layerwise/Meta-Llama-3.1-8B-msmarco-mntp-L16/checkpoint-3930 --output_dir sparse_dumps"

# PKQ_GPU_NUM=0
# PKQ_RUN_COMMAND="python experiments/fuse_eval.py --dump_dir ${PKQ_DATA_DIR}/backup/20261011_113330/sparse_dumps ${PKQ_DATA_DIR}/backup/20261009_105202/sparse_dumps --layers 16 26 --output_dir results_fusion"

# PKQ_RUN_COMMAND="python experiments/mteb_eval_splade.py --output_dir results_splade_v3_nopool"

# PKQ_RUN_COMMAND="python experiments/check_postprocessing.py $PKQ_DATA_DIR/Llama3_1-8B-Base-L0R-8x"

# PKQ_RUN_COMMAND="python experiments/diag_sae.py"

# PKQ_RUN_COMMAND="python experiments/check_sae_threshold.py \
#     --sae_weights_path ${PKQ_DATA_DIR}/Llama3_1-8B-Base-L0R-8x/checkpoints/final.safetensors"


# PKQ_RUN_COMMAND="torchrun --nproc_per_node=${PKQ_GPU_NUM} experiments/run_multilayer_finetune.py --config train_configs/multilayer/MetaLlama3.1-mntp-L16_26.json --hard_negatives_file /proj/berzelius-aiics-real/users/x_jinma/project_remote_pkq/remote_data/llm2vec/reranker_parts/"


# PKQ_RUN_COMMAND="python experiments/mteb_eval_multilayer.py --trained_checkpoint_path ${PKQ_DATA_DIR}/backup/20261012_093224/output/multilayer/Meta-Llama-3.1-8B-msmarco-mntp-L16+26/checkpoint-3930 --task_name SciFact NFCorpus ArguAna SCIDOCS --output_dir results_mteb_multilayer"


# PKQ_RUN_COMMAND="torchrun --nproc_per_node=4 experiments/run_docside_finetune.py --config train_configs/docside/MetaLlama3.1-mntp-docside.json --hard_negatives_file ${PKQ_DATA_DIR}/reranker_parts/"


# PKQ_RUN_COMMAND="python experiments/mteb_eval_docside.py --trained_checkpoint_path ${PKQ_DATA_DIR}/backup/20261013_093756/output/docside/Meta-Llama-3.1-8B-msmarco-mntp-L16/checkpoint-3930 --task_name SciFact NFCorpus ArguAna SCIDOCS --output_dir results_mteb_docside"


# PKQ_RUN_COMMAND="python experiments/sae_inject_qa.py --trained_checkpoint_path ${PKQ_DATA_DIR}/backup/20261010_125818/output/layerwise/Meta-Llama-3.1-8B-msmarco-mntp-L16/checkpoint-3930 --output_dir results_sae_inject"

# PKQ_RUN_COMMAND="python experiments/feature_drift.py --trained_checkpoint_path ${PKQ_DATA_DIR}/backup/20261010_125818/output/layerwise/Meta-Llama-3.1-8B-msmarco-mntp-L16/checkpoint-3930"

# PKQ_RUN_COMMAND="python experiments/feature_contexts.py --trained_checkpoint_path ${PKQ_DATA_DIR}/backup/20261010_125818/output/layerwise/Meta-Llama-3.1-8B-msmarco-mntp-L16/checkpoint-3930"

# PKQ_RUN_COMMAND="python experiments/feature_lexical.py --trained_checkpoint_path ${PKQ_DATA_DIR}/backup/20261010_125818/output/layerwise/Meta-Llama-3.1-8B-msmarco-mntp-L16/checkpoint-3930"

# PKQ_RUN_COMMAND="torchrun --nproc_per_node=4 experiments/run_featgate_finetune.py --config train_configs/featgate/MetaLlama3.1-mntp-featgate.json --hard_negatives_file /proj/berzelius-aiics-real/users/x_jinma/project_remote_pkq/remote_data/llm2vec/reranker_parts/"

# PKQ_RUN_COMMAND="python experiments/mteb_eval_featgate.py --trained_checkpoint_path ${PKQ_DATA_DIR}/backup/20261013_155049/output/featgate/Meta-Llama-3.1-8B-msmarco-mntp-L16/checkpoint-3930 --task_name SciFact NFCorpus ArguAna SCIDOCS"

# PKQ_RUN_COMMAND="python experiments/mteb_eval_featgate.py --trained_checkpoint_path /proj/berzelius-aiics-real/users/x_jinma/project_remote_pkq/remote_data/llm2vec/backup/20261013_155049/output/featgate/Meta-Llama-3.1-8B-msmarco-mntp-L16/checkpoint-3930 --task_name SciFact NFCorpus ArguAna SCIDOCS --no_gate"

# PKQ_RUN_COMMAND="python experiments/feature_lexical_featgate.py --trained_checkpoint_path ${PKQ_DATA_DIR}/backup/20261010_125818/output/layerwise/Meta-Llama-3.1-8B-msmarco-mntp-L16/checkpoint-3930 --featgate_path ${PKQ_DATA_DIR}/backup/20261013_155049/output/featgate/Meta-Llama-3.1-8B-msmarco-mntp-L16/checkpoint-3930/featgate.pt"

# PKQ_RUN_COMMAND="torchrun --nproc_per_node=4 experiments/run_toplayers_finetune.py --config train_configs/toplayers/MetaLlama3.1-mntp-toplayers.json --hard_negatives_file ${PKQ_DATA_DIR}/reranker_parts/"


# PKQ_RUN_COMMAND="python experiments/mteb_eval_layerwise.py --trained_checkpoint_path ${PKQ_DATA_DIR}/backup/20261013_195353/output/toplayers/Meta-Llama-3.1-8B-msmarco-mntp-L16-from13/checkpoint-3930 --task_name SciFact NFCorpus ArguAna SCIDOCS"

# PKQ_RUN_COMMAND="python experiments/feature_lexical.py --trained_checkpoint_path ${PKQ_DATA_DIR}/backup/20261013_195353/output/toplayers/Meta-Llama-3.1-8B-msmarco-mntp-L16-from13/checkpoint-3930"

# PKQ_RUN_COMMAND="torchrun --nproc_per_node=4 experiments/run_anchor_finetune.py --config train_configs/anchor/MetaLlama3.1-mntp-anchor.json --hard_negatives_file ${PKQ_DATA_DIR}/reranker_parts/"

# PKQ_RUN_COMMAND="python experiments/mteb_eval_layerwise.py --trained_checkpoint_path ${PKQ_DATA_DIR}/backup/20261014_203212/output/anchor/Meta-Llama-3.1-8B-msmarco-mntp-L16-from13-anchor1/checkpoint-3930 --task_name SciFact NFCorpus ArguAna SCIDOCS"

# PKQ_RUN_COMMAND="python experiments/feature_lexical.py --trained_checkpoint_path ${PKQ_DATA_DIR}/backup/20261014_203212/output/anchor/Meta-Llama-3.1-8B-msmarco-mntp-L16-from13-anchor1/checkpoint-3930"

# PKQ_RUN_COMMAND="python experiments/feature_drift.py --trained_checkpoint_path ${PKQ_DATA_DIR}/backup/20261014_203212/output/anchor/Meta-Llama-3.1-8B-msmarco-mntp-L16-from13-anchor1/checkpoint-3930"


# PKQ_RUN_COMMAND="python experiments/feature_drift.py --trained_checkpoint_path ${PKQ_DATA_DIR}/backup/20261013_195353/output/toplayers/Meta-Llama-3.1-8B-msmarco-mntp-L16-from13/checkpoint-3930"


# PKQ_RUN_COMMAND="torchrun --nproc_per_node=4 experiments/run_anchor_finetune.py --config train_configs/anchor_sweep/MetaLlama3.1-mntp-anchor0.1.json --hard_negatives_file ${PKQ_DATA_DIR}/reranker_parts/"


# PKQ_RUN_COMMAND="python experiments/mteb_eval_layerwise.py --trained_checkpoint_path ${PKQ_DATA_DIR}/backup/20261015_171125/output/anchor/Meta-Llama-3.1-8B-msmarco-mntp-L16-from13-anchor0.1/checkpoint-3930 --task_name SciFact NFCorpus ArguAna SCIDOCS"

# PKQ_RUN_COMMAND="python experiments/feature_lexical.py --trained_checkpoint_path ${PKQ_DATA_DIR}/backup/20261015_171125/output/anchor/Meta-Llama-3.1-8B-msmarco-mntp-L16-from13-anchor0.1/checkpoint-3930"

# PKQ_RUN_COMMAND="torchrun --nproc_per_node=4 experiments/run_anchor_finetune.py --config train_configs/anchor_sweep/MetaLlama3.1-mntp-anchor3.json --hard_negatives_file ${PKQ_DATA_DIR}/reranker_parts/"


# PKQ_RUN_COMMAND="python experiments/mteb_eval_layerwise.py --trained_checkpoint_path ${PKQ_DATA_DIR}/backup/20261015_210817/output/anchor/Meta-Llama-3.1-8B-msmarco-mntp-L16-from13-anchor3/checkpoint-3930 --task_name SciFact NFCorpus ArguAna SCIDOCS"

# PKQ_RUN_COMMAND="python experiments/feature_lexical.py --trained_checkpoint_path ${PKQ_DATA_DIR}/backup/20261015_210817/output/anchor/Meta-Llama-3.1-8B-msmarco-mntp-L16-from13-anchor3/checkpoint-3930"

# PKQ_RUN_COMMAND="python experiments/mteb_eval_layerwise.py --trained_checkpoint_path ${PKQ_DATA_DIR}/backup/20261013_195353/output/toplayers/Meta-Llama-3.1-8B-msmarco-mntp-L16-from13/checkpoint-3930 --task_name SciFact NFCorpus ArguAna SCIDOCS"


# PKQ_RUN_COMMAND="python experiments/mteb_eval_layerwise.py --trained_checkpoint_path ${PKQ_DATA_DIR}/backup/20261015_171125/output/anchor/Meta-Llama-3.1-8B-msmarco-mntp-L16-from13-anchor0.1/checkpoint-3930 --task_name SciFact NFCorpus ArguAna SCIDOCS"

# PKQ_RUN_COMMAND="python experiments/mteb_eval_layerwise.py --trained_checkpoint_path ${PKQ_DATA_DIR}/backup/20261014_203212/output/anchor/Meta-Llama-3.1-8B-msmarco-mntp-L16-from13-anchor1/checkpoint-3930 --task_name SciFact NFCorpus ArguAna SCIDOCS"
#
# PKQ_RUN_COMMAND="python experiments/mteb_eval_layerwise.py --trained_checkpoint_path ${PKQ_DATA_DIR}/backup/20261015_210817/output/anchor/Meta-Llama-3.1-8B-msmarco-mntp-L16-from13-anchor3/checkpoint-3930 --task_name SciFact NFCorpus ArguAna SCIDOCS"
#
# PKQ_RUN_COMMAND="python experiments/mteb_eval_featgate.py --trained_checkpoint_path ${PKQ_DATA_DIR}/backup/20261013_155049/output/featgate/Meta-Llama-3.1-8B-msmarco-mntp-L16/checkpoint-3930 --task_name SciFact NFCorpus ArguAna SCIDOCS"
#
# PKQ_RUN_COMMAND="python experiments/mteb_eval_featgate.py --trained_checkpoint_path ${PKQ_DATA_DIR}/backup/20261013_155049/output/featgate/Meta-Llama-3.1-8B-msmarco-mntp-L16/checkpoint-3930 --task_name SciFact NFCorpus ArguAna SCIDOCS --no_gate"

# PKQ_RUN_COMMAND="python experiments/feature_lexical_basew.py --trained_checkpoint_path ${PKQ_DATA_DIR}/backup/20261015_210817/output/anchor/Meta-Llama-3.1-8B-msmarco-mntp-L16-from13-anchor3/checkpoint-3930"

# PKQ_RUN_COMMAND="python experiments/feature_lexical_basew.py --trained_checkpoint_path ${PKQ_DATA_DIR}/backup/20261014_203212/output/anchor/Meta-Llama-3.1-8B-msmarco-mntp-L16-from13-anchor1/checkpoint-3930"
#
# PKQ_RUN_COMMAND="python experiments/feature_lexical_basew.py --trained_checkpoint_path ${PKQ_DATA_DIR}/backup/20261015_171125/output/anchor/Meta-Llama-3.1-8B-msmarco-mntp-L16-from13-anchor0.1/checkpoint-3930"
#
# PKQ_RUN_COMMAND="python experiments/feature_lexical_basew.py --trained_checkpoint_path ${PKQ_DATA_DIR}/backup/20261013_195353/output/toplayers/Meta-Llama-3.1-8B-msmarco-mntp-L16-from13/checkpoint-3930"
#
# PKQ_RUN_COMMAND="python experiments/feature_lexical_basew.py --trained_checkpoint_path ${PKQ_DATA_DIR}/backup/20261010_125818/output/layerwise/Meta-Llama-3.1-8B-msmarco-mntp-L16/checkpoint-3930"


# PKQ_RUN_COMMAND="python experiments/feature_lexical_featgate_basew.py --trained_checkpoint_path ${PKQ_DATA_DIR}/backup/20261010_125818/output/layerwise/Meta-Llama-3.1-8B-msmarco-mntp-L16/checkpoint-3930 --featgate_path ${PKQ_DATA_DIR}/backup/20261013_155049/output/featgate/Meta-Llama-3.1-8B-msmarco-mntp-L16/checkpoint-3930/featgate.pt"

# PKQ_RUN_COMMAND="python experiments/feature_precision.py --trained_checkpoint_path ${PKQ_DATA_DIR}/backup/20261010_125818/output/layerwise/Meta-Llama-3.1-8B-msmarco-mntp-L16/checkpoint-3930 --featgate_path ${PKQ_DATA_DIR}/backup/20261013_155049/output/featgate/Meta-Llama-3.1-8B-msmarco-mntp-L16/checkpoint-3930/featgate.pt"
# PKQ_RUN_COMMAND="python experiments/feature_precision.py --trained_checkpoint_path ${PKQ_DATA_DIR}/backup/20261013_195353/output/toplayers/Meta-Llama-3.1-8B-msmarco-mntp-L16-from13/checkpoint-3930"
# PKQ_RUN_COMMAND="python experiments/feature_precision.py --trained_checkpoint_path ${PKQ_DATA_DIR}/backup/20261015_171125/output/anchor/Meta-Llama-3.1-8B-msmarco-mntp-L16-from13-anchor0.1/checkpoint-3930"
# PKQ_RUN_COMMAND="python experiments/feature_precision.py --trained_checkpoint_path ${PKQ_DATA_DIR}/backup/20261014_203212/output/anchor/Meta-Llama-3.1-8B-msmarco-mntp-L16-from13-anchor1/checkpoint-3930"
# PKQ_RUN_COMMAND="python experiments/feature_precision.py --trained_checkpoint_path ${PKQ_DATA_DIR}/backup/20261015_210817/output/anchor/Meta-Llama-3.1-8B-msmarco-mntp-L16-from13-anchor3/checkpoint-3930"

# PKQ_RUN_COMMAND="torchrun --nproc_per_node=${PKQ_GPU_NUM} experiments/feature_scale_base.py"

# PKQ_RUN_COMMAND="torchrun --nproc_per_node=${PKQ_GPU_NUM} experiments/feature_scale_base_fvu.py"

# PKQ_RUN_COMMAND="torchrun --nproc_per_node=${PKQ_GPU_NUM} experiments/label_snippets.py --trained_checkpoint_path ${PKQ_DATA_DIR}/backup/20261010_125818/output/layerwise/Meta-Llama-3.1-8B-msmarco-mntp-L16/checkpoint-3930 --name splare"

# PKQ_RUN_COMMAND="torchrun --nproc_per_node=${PKQ_GPU_NUM} experiments/label_judge.py --snippets ${PKQ_DATA_DIR}/backup/20261027_152412/results_label_snippets/label_snippets_splare.json"

# PKQ_RUN_COMMAND="torchrun --nproc_per_node=${PKQ_GPU_NUM} experiments/label_snippets.py --trained_checkpoint_path ${PKQ_DATA_DIR}/backup/20261013_195353/output/toplayers/Meta-Llama-3.1-8B-msmarco-mntp-L16-from13/checkpoint-3930 --name top-layers"

# PKQ_RUN_COMMAND="torchrun --nproc_per_node=${PKQ_GPU_NUM} experiments/label_snippets.py --trained_checkpoint_path ${PKQ_DATA_DIR}/backup/20261014_203212/output/anchor/Meta-Llama-3.1-8B-msmarco-mntp-L16-from13-anchor1/checkpoint-3930 --name anchor"

# PKQ_RUN_COMMAND="torchrun --nproc_per_node=${PKQ_GPU_NUM} experiments/label_judge.py --snippets ${PKQ_DATA_DIR}/backup/20261028_110410/results_label_snippets/label_snippets_top-layers.json"

# PKQ_RUN_COMMAND="torchrun --nproc_per_node=${PKQ_GPU_NUM} experiments/label_judge.py --snippets ${PKQ_DATA_DIR}/backup/20261029_151719/results_label_snippets/label_snippets_anchor.json"

PKQ_GPU_NUM=1
PKQ_RUN_COMMAND="python experiments/fusion_search.py --dump_dir ${PKQ_DATA_DIR}/backup/20261011_113330/sparse_dumps ${PKQ_DATA_DIR}/backup/20261009_105202/sparse_dumps --layers 16 26 --output_dir results_fusion_search"

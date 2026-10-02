
set -e
# set following vars based on your setting, and then make sure this repo is cloned to /proj/berzelius-aiics-real/users/x_jinma/project_remote_pkq/llm2vec_pikaq
export RUN_DIR_HOME=
export RUN_PROJ=

cd ${RUN_DIR_HOME}/project_remote_pkq/${RUN_PROJ}
export PKQ_PYTHON="3.10"
export PKQ_SERVER_NAME=berzeliusampere
export PKQ_MODULES="Miniforge3 buildenv-gcccuda/12.4.1-gcc13.3.0"
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
export PKQ_RUN_COMMAND="torchrun --nproc_per_node=${PKQ_GPU_NUM} experiments/run_layerwise_finetune.py     --config train_configs/layerwise/MetaLlama3.1-mntp-layerwise.json     --hard_negatives_file ${PKQ_DATA_DIR}/reranker_parts/"
if [ -z  ]; then
    export PKQ_CONDAENV=/proj/berzelius-aiics-real/users/x_jinma/pkqcondaenv/llm2vec_pikaq
    export PKQ_WHEELS=/proj/berzelius-aiics-real/users/x_jinma/pkqwheels/llm2vec_pikaq
fi
echo "condaenv path "
sbatch --signal=B:USR1@120 --time=1-00:00:00 --nodes=1 --output=pkqlogs/20261002_194252/job-%j.out --error=pkqlogs/20261002_194252/job-%j.out --nodelist=node[061-064,065,066-093] --gpus=4 --cpus-per-task=32 --mem=96G  -A berzelius-2026-243  --partition=berzelius pkq_configs/remote/remote_tmps/slurm.sh

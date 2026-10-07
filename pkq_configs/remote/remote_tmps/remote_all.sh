
set -e
# set following vars based on your setting, and then make sure this repo is cloned to /nobackup/proj/disk/naiss2026-3-658/personal/jinma63/project_remote_pkq/llm2vec_pikaq
export RUN_DIR_HOME=
export RUN_PROJ=

cd ${RUN_DIR_HOME}/project_remote_pkq/${RUN_PROJ}
export PKQ_PYTHON="3.10"
export PKQ_SERVER_NAME=arrhenius
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
if [[ -z ${PKQ_MODULES} ]]; then
export PKQ_MODULES="GPU/Miniforge/26.3.2-2-eb"
fi
export PKQ_ARCH="aarch64"

if [ -z ${PKQ_CONDAENV} ]; then
    export PKQ_CONDAENV=${RUN_DIR_HOME}/pkqcondaenv/${RUN_PROJ}
    export PKQ_WHEELS=${RUN_DIR_HOME}/pkqwheels/${RUN_PROJ}
fi
echo "condaenv path ${PKQ_CONDAENV}"
sbatch --signal=B:USR1@120 --time=1-00:00:00 --nodes=1 --output=pkqlogs/20261007_103800/job-%j.out --error=pkqlogs/20261007_103800/job-%j.out  --gres=gpu:4 --cpus-per-task=32 --mem=320G  -A naiss2026-3-658-gpu  --partition=gpu pkq_configs/remote/remote_tmps/slurm.sh


set -e
# change the following vars based on your preference, and then make sure this repo is cloned to /nobackup/proj/disk/naiss2026-3-658/personal/jinma63/project_remote_pkq/llm2vec_pikaq
export RUN_DIR_HOME=/nobackup/proj/disk/naiss2026-3-658/personal/jinma63
export RUN_PROJ=llm2vec_pikaq
export PKQ_DATA_DIR=/nobackup/proj/disk/naiss2026-3-658/personal/jinma63/project_remote_pkq/remote_data/llm2vec

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
export PKQ_ARCH="aarch64"
export PKQ_MODULES="GPU/Miniforge/26.3.2-2-eb GPU/buildtool-easybuild/5.2.1-hpca3ef7d197 CUDA/12.9.1 cuDNN/9.15.0.57-CUDA-12.9.1 cuSPARSELt/0.8.0.4-CUDA-12.9.1"
if [ -z  ]; then
    export PKQ_CONDAENV=/nobackup/proj/disk/naiss2026-3-658/personal/jinma63/pkqcondaenv/llm2vec_pikaq
    export PKQ_WHEELS=/nobackup/proj/disk/naiss2026-3-658/personal/jinma63/pkqwheels/llm2vec_pikaq
fi
echo "condaenv path "
sbatch --signal=B:USR1@120 --time=1-00:00:00 --nodes=1 --output=pkqlogs/20260921_112134/job-%j.out --error=pkqlogs/20260921_112134/job-%j.out  --gres=gpu:4 --cpus-per-task=32 --mem=320G  -A naiss2026-3-658-gpu  --partition=gpu pkq_configs/remote/remote_tmps/slurm.sh

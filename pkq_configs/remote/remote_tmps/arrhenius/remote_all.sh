
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
export PKQ_TASK_NAMES="ArguAna"
export PKQ_RUN_COMMAND="python experiments/mteb_eval_layerwise.py --trained_checkpoint_path ${PKQ_DATA_DIR}/backup/arrhenius/20260926_074205/output/layerwise/Meta-Llama-3.1-8B-msmarco-mntp-L26/checkpoint-3930 --query_top_k 40 --doc_top_k 400 --output_dir results --max_length 1024 --task_name ${PKQ_TASK_NAMES}"
if [[ -z ${PKQ_MODULES} ]]; then
export PKQ_MODULES="GPU/Miniforge/26.3.2-2-eb"
fi
export PKQ_ARCH="aarch64"

if [ -z ${PKQ_CONDAENV} ]; then
    export PKQ_CONDAENV=${RUN_DIR_HOME}/pkqcondaenv/${RUN_PROJ}
    export PKQ_WHEELS=${RUN_DIR_HOME}/pkqwheels/${RUN_PROJ}
fi
echo "condaenv path ${PKQ_CONDAENV}"
module --force purge
module load ${PKQ_MODULES}

if [[ ! -d ${PKQ_CONDAENV}${PKQ_ARCH} ]]; then
    conda create -p ${PKQ_CONDAENV}${PKQ_ARCH} python=${PKQ_PYTHON} pip -y
fi
conda activate ${PKQ_CONDAENV}${PKQ_ARCH}
which python
python --version
which pip
# pip install -e .
# pip install torch --force-reinstall --index-url https://download.pytorch.org/whl/cu128
# pip install ninja
# pip install datasets==3.6.0
# pip install seqeval
# pip install jupyterlab
# pip install sentence_transformers
# pip install sentencepiece
# pip install protobuf
#
# pip install peft==0.12.0
#
# pip install mteb
# pip install ir_datasets
# pip install -q huggingface_hub
# pip install ijson
#
# pip install --force-reinstall transformers==4.44.2
#
#
# hf download Tevatron/msmarco-passage-corpus --repo-type dataset
#
# hf download "meta-llama/Meta-Llama-3.1-8B" --exclude "*.pth"
# hf download "OpenMOSS-Team/Llama3_1-8B-Base-LXR-8x" \
#   --include "Llama3_1-8B-Base-L26R-8x/*"
# hf download "OpenMOSS-Team/Llama3_1-8B-Base-LXR-8x" \
#   --include "Llama3_1-8B-Base-L0R-8x/*"
# hf download "naver/trecdl22-crossencoder-debertav3"
#
# hf download OpenMOSS-Team/Llama3_1-8B-Base-LXR-32x --include "Llama3_1-8B-Base-L26R-32x/*"
# hf download naver/splade-v3
pip list > pkq_configs/packages.txt
sbatch --signal=B:USR1@120 --time=1-00:00:00 --nodes=1 --output=pkqlogs/20261008_131347/job-%j.out --error=pkqlogs/20261008_131347/job-%j.out  --gres=gpu:4 --cpus-per-task=32 --mem=320G  -A naiss2026-3-658-gpu --partition=gpu pkq_configs/remote/remote_tmps/arrhenius/slurm.sh

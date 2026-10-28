
set -e
# set following vars based on your setting, and then make sure this repo is cloned to /proj/berzelius-aiics-real/users/x_jinma/project_remote_pkq/llm2vec_pikaq
export RUN_DIR_HOME=
export RUN_PROJ=

cd ${RUN_DIR_HOME}/project_remote_pkq/${RUN_PROJ}
export PKQ_PYTHON="3.10"
export PKQ_SERVER_NAME=berzeliushopper
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
export PKQ_RUN_COMMAND="torchrun --nproc_per_node=${PKQ_GPU_NUM} experiments/label_snippets.py --trained_checkpoint_path ${PKQ_DATA_DIR}/backup/20261010_125818/output/layerwise/Meta-Llama-3.1-8B-msmarco-mntp-L16/checkpoint-3930 --name splare"
if [[ -z ${PKQ_MODULES} ]]; then
export PKQ_MODULES="Miniforge3"
fi
export PKQ_LOGIN_MODULES=${PKQ_MODULES}
module --force purge
module load ${PKQ_LOGIN_MODULES}
PKQTMP=${RUN_DIR_HOME}/pkqcondaenv/pkqbase
if [[ ! -d ${PKQTMP} ]]; then
    conda create -p ${PKQTMP} pip -y
fi
conda activate ${PKQTMP}
which python
python --version
which pip
pip install -q huggingface_hub
if [ -z ${PKQ_CONDAENV} ]; then
    export PKQ_CONDAENV=${RUN_DIR_HOME}/pkqcondaenv/${RUN_PROJ}
    export PKQ_WHEELS=${RUN_DIR_HOME}/pkqwheels/${RUN_PROJ}
fi
echo "condaenv path ${PKQ_CONDAENV}"
set -euo pipefail

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

# hf download OpenMOSS-Team/Llama3_1-8B-Base-LXR-32x --include "Llama3_1-8B-Base-L0R-32x/*"

# hf download OpenMOSS-Team/Llama3_1-8B-Base-LXR-32x --include "Llama3_1-8B-Base-L16R-32x/*"

# Neuronpedia labels (explanations) of the Llama Scope L16R-32x SAE (public S3 export, not on HF)
# NP_URL=https://neuronpedia-datasets.s3.us-east-1.amazonaws.com/v1/llama3.1-8b/16-llamascope-res-131k
# NP_DIR=${PKQ_DATA_DIR}/neuronpedia/16-llamascope-res-131k
# mkdir -p ${NP_DIR}/explanations
# curl -fsS -o ${NP_DIR}/source.jsonl ${NP_URL}/source.jsonl
# i=0
# while true; do
#   code=$(curl -sS -o /dev/null -w '%{http_code}' -I ${NP_URL}/explanations/batch-${i}.jsonl.gz)
#   if [ "${code}" = 404 ]; then break; fi                  # past the last batch
#   if [ "${code}" != 200 ]; then echo "batch-${i}: HTTP ${code}" >&2; exit 1; fi
#   curl -fsS -o ${NP_DIR}/explanations/batch-${i}.jsonl.gz ${NP_URL}/explanations/batch-${i}.jsonl.gz
#   i=$((i + 1))
# done
# echo "downloaded ${i} explanation batches to ${NP_DIR}/explanations"

# judge LLM for the label check
hf download Qwen/Qwen2.5-72B-Instruct

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

pip list > pkq_configs/remote/remote_tmps/berzeliushopper/packages.txt && exit
sbatch --signal=B:USR1@120 --time=1-00:00:00 --nodes=1 --output=pkq_configs/logs_20261027_152412/job-%j.out --error=pkq_configs/logs_20261027_152412/job-%j.out  --gpus=4 --cpus-per-task=32 --mem=96G  -A berzelius-2026-243 --partition=berzelius-hopper pkq_configs/remote/remote_tmps/berzeliushopper/slurm.sh

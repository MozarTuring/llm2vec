if [[ -z ${PKQ_MODULES} ]]; then
export PKQ_MODULES="GPU/Miniforge/26.3.2-2-eb"
fi
export PKQ_LOGIN_MODULES="Miniforge/26.3.2-2-eb"
export PKQ_ARCH="aarch64"
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

# NP_URL=https://neuronpedia-datasets.s3.us-east-1.amazonaws.com/v1/llama3.1-8b/16-llamascope-res-131k
# NP_DIR=./neuronpedia/16-llamascope-res-131k
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

# hf download Qwen/Qwen2.5-72B-Instruct
hf download Qwen/Qwen2.5-32B-Instruct


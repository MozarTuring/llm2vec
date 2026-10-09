if [[ -z ${PKQ_MODULES} ]]; then
export PKQ_MODULES="GPU/Miniforge/26.3.2-2-eb"
fi
export PKQ_LOGIN_MODULES="Miniforge/26.3.2-2-eb"
export PKQ_ARCH="aarch64"

if [ -z ${PKQ_CONDAENV} ]; then
    export PKQ_CONDAENV=${RUN_DIR_HOME}/pkqcondaenv/${RUN_PROJ}
    export PKQ_WHEELS=${RUN_DIR_HOME}/pkqwheels/${RUN_PROJ}
fi
echo "condaenv path ${PKQ_CONDAENV}"
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

hf download OpenMOSS-Team/Llama3_1-8B-Base-LXR-32x --include "Llama3_1-8B-Base-L0R-32x/*"

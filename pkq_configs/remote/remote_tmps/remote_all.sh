
set -e
# change the following vars based on your preference, and then make sure this repo is cloned to /nobackup/proj/disk/naiss2026-3-658/personal/jinma63/project_remote_jwm/llm2vec_jingwei
export RUN_DIR_HOME=/nobackup/proj/disk/naiss2026-3-658/personal/jinma63
export RUN_PROJ=llm2vec_jingwei
export JWM_DATA_DIR=/nobackup/proj/disk/naiss2026-3-658/personal/jinma63/project_remote_jwm/remote_data/llm2vec

cd ${RUN_DIR_HOME}/project_remote_jwm/${RUN_PROJ}
export JWM_PYTHON="3.10"
export JWM_SERVER_NAME=arrhenius
export JWM_GPU_NUM=1
export JWM_NODES_NUM=1
export JWM_RUN_TIME="1-00:00:00"
export JWM_SLURM_FILE=slurm.sh
export JWM_build_flashattn=
export JWM_NOTEBOOK=
export MEM_PER_TASK="$((30 * JWM_GPU_NUM))G"
export CPUS_PER_TASK=$((8 * JWM_GPU_NUM))
export CUDA_VISIBLE_DEVICES=1
export JWM_TASK_NAMES="ArguAna CQADupstackGamingRetrieval CQADupstackUnixRetrieval ClimateFEVERHardNegatives FEVERHardNegatives FiQA2018 HotpotQAHardNegatives SCIDOCS TRECCOVID Touche2020Retrieval.v3"
export JWM_RUN_COMMAND="python experiments/mteb_eval_layerwise.py   --trained_checkpoint_path ${JWM_DATA_DIR}/output/layerwise/Meta-Llama-3.1-8B-msmarco-mntp-L0/checkpoint-3929   --query_top_k 40   --doc_top_k 400   --output_dir results   --max_length 1024   --task_name ${JWM_TASK_NAMES}"
export JWM_ARCH="aarch64"
export JWM_MODULES="Miniforge"
module load ${JWM_MODULES}
if [ -z ${JWM_CONDAENV} ]; then
    export JWM_CONDAENV=${RUN_DIR_HOME}/jwmcondaenv/${RUN_PROJ}
    export JWM_WHEELS=${RUN_DIR_HOME}/jwmwheels/${RUN_PROJ}
fi
echo "condaenv path ${JWM_CONDAENV}"
if [ ! -d ${JWM_CONDAENV} ]; then
    conda create -p ${JWM_CONDAENV} python=${JWM_PYTHON} -y
fi
if [[ -n ${JWM_ARCH} && ! -d ${JWM_CONDAENV}${JWM_ARCH} ]]; then
    CONDA_SUBDIR=linux-aarch64 conda create -p ${JWM_CONDAENV}${JWM_ARCH} python=${JWM_PYTHON} -y
fi
conda activate ${JWM_CONDAENV}
which python
python --version
which pip
## Download aarch64 wheels on login node (x86_64) for offline install on compute nodes

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

# pip install mteb
# pip install ir_datasets
# pip install -q huggingface_hub
# pip install ijson

if [[ "${JWM_ARCH}" == "aarch64" ]]; then
    export _wheel_dir=${JWM_WHEELS}${JWM_ARCH}
    PLATFORM="manylinux2014_aarch64"

    mkdir -p "${_wheel_dir}"

    pip download \
        --platform "${PLATFORM}" \
        --python-version "${JWM_PYTHON}" \
        --only-binary=:all: \
        -d "${_wheel_dir}" \
        torch --index-url https://download.pytorch.org/whl/cu128

    pip download \
        --platform "${PLATFORM}" \
        --python-version "${JWM_PYTHON}" \
        --only-binary=:all: \
        -d "${_wheel_dir}" \
        ninja \
        "datasets==3.6.0" \
        seqeval \
        jupyterlab \
        sentence_transformers \
        sentencepiece \
        protobuf \
        "peft==0.12.0" \
        mteb \
        ir_datasets \
        huggingface_hub \
        ijson \
        tomli

fi

# python experiments/download_model.py \
#     --model_name_or_path meta-llama/Meta-Llama-3.1-8B \
#     --dataset_name Tevatron/msmarco-passage-corpus

# hf download "OpenMOSS-Team/Llama3_1-8B-Base-LXR-8x" \
#   --include "Llama3_1-8B-Base-L26R-8x/*" \
#   --local-dir "${JWM_DATA_DIR}"

# hf download "naver/trecdl22-crossencoder-debertav3" \
#   --include "*" \
#   --local-dir "${JWM_DATA_DIR}/hf_models/naver/trecdl22-crossencoder-debertav3"
sbatch --signal=B:USR1@120 --time=1-00:00:00 --nodes=1 --output=jwmlogs/20260918_082342/job-%j.out --error=jwmlogs/20260918_082342/job-%j.out  --gpus=1 --cpus-per-task=8 --mem=30G  -A naiss2026-3-658-gpu  --partition=gpu jwm_configs/remote/remote_tmps/slurm.sh

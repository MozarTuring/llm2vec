set -e 
export JWM_GPU_NUM=1
export JWM_SLURM_FILE=slurm.sh
export JWM_NODES_NUM=1
export JWM_SERVER_NAME=berzeliusampere
export JWM_RUN_COMMAND="python experiments/run_layerwise_finetune.py train_configs/layerwise/MetaLlama3.1-mntp-layerwise.json"
export JWM_build_flashattn=
export JWM_NOTEBOOK=
export JWM_RUN_TIME="1-00:00:00"

require_env() {
for var in "$@"; do
    if [ -z "${!var}" ]; then
        echo "Error: $var is not set" >&2
        exit 1
    fi
done
}

export PYTHONUNBUFFERED=1
# change the following based on your running preference
export RUN_DIR_HOME="/home/x_jinma"
export RUN_PROJ="llm2vec_jingwei"

export JWM_MODULES="Miniforge3 buildenv-gcccuda/12.4.1-gcc13.3.0"
export JWM_CONDAENV="${RUN_DIR_HOME}/conda_envs/llm2vec"
if [[ ${JWM_SLURM_RUN_ARGS} == *"MetaLlama3"* ]]; then

    export JWM_SLURM_NODES="--nodelist=node[061-064,065,066-093]"
fi
if [[ -n ${JWM_build_flashattn} ]]; then
    export CPUS_PER_TASK=32
    export MEM_PER_TASK="256G"
else
    export CPUS_PER_TASK=$((8 * JWM_GPU_NUM))
    export MEM_PER_TASK="$((24 * JWM_GPU_NUM))G"
fi

module --force purge
module load ${JWM_MODULES}

if [ ! -d ${JWM_CONDAENV} ]; then

    conda create -p ${JWM_CONDAENV} python=3.10 -y

fi
conda activate ${JWM_CONDAENV}
which python
echo $PWD


# pip install -e .
# pip install torch --force-reinstall --index-url https://download.pytorch.org/whl/cu128
# pip install ninja
# pip uninstall -y flash-attn 2>/dev/null
# mkdir -p ${JWM_CONDAENV}/flash_attn_src
# python -c "
# import json, urllib.request, os
# dest = os.environ['JWM_CONDAENV'] + '/flash_attn_src/flash_attn-2.8.3.post1.tar.gz'
# if not os.path.exists(dest):
#     data = json.loads(urllib.request.urlopen('https://pypi.org/pypi/flash-attn/2.8.3.post1/json').read())
#     url = [u['url'] for u in data['urls'] if u['packagetype'] == 'sdist'][0]
#     print(f'Downloading {url}')
#     urllib.request.urlretrieve(url, dest)
#     print('Done')
# else:
#     print('Source tarball already exists')
# "

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
#
# python experiments/download_model.py \
#     --model_name_or_path meta-llama/Meta-Llama-3.1-8B \
#     --dataset_name Tevatron/msmarco-passage-corpus


# hf download "OpenMOSS-Team/Llama3_1-8B-Base-LXR-8x" \
#   --include "Llama3_1-8B-Base-L26R-8x/*" \
#   --local-dir "${JWM_DATA_DIR}"

require_env JWM_SLURM_FILE JWM_RUN_TIME JWM_NODES_NUM
if [[ ${JWM_NOTEBOOK} == 1 ]];then
    JWM_RUN_COMMAND="jupyter lab --MappingKernelManager.cull_idle_timeout=3600 --MappingKernelManager.cull_interval=360 --MappingKernelManager.cull_connected=True --ip=0.0.0.0 --port=18889 --no-browser --allow-root --NotebookApp.token=''"
    JWM_SLURM_RUN_ARGS=""
fi
cat ${RUN_DIR_HOME}/project_remote_jwm/common_tools_jingwei/slurm_header.sh ${JWM_SLURM_FILE} > jwm_configs/${JWM_MODE}/remote_tmps/${JWM_SLURM_FILE}

echo "

export LD_LIBRARY_PATH=${LIBRARY_PATH}:${LD_LIBRARY_PATH:-}

${JWM_RUN_COMMAND} &
wait \$!
" >> jwm_configs/${JWM_MODE}/remote_tmps/${JWM_SLURM_FILE}

sbatch_args="--signal=B:USR1@120 --time=${JWM_RUN_TIME} --nodes=${JWM_NODES_NUM} --output=jwmlogs/${JWM_RUN_START_TIME}/job-%j.out --error=jwmlogs/${JWM_RUN_START_TIME}/job-%j.out ${JWM_SLURM_NODES}"
sbatch_args="${sbatch_args} --gpus=${JWM_GPU_NUM} --cpus-per-task=${CPUS_PER_TASK} --mem=${MEM_PER_TASK}  -A berzelius-2026-50 --partition=berzelius"

echo ${sbatch_args} jwm_configs/${JWM_MODE}/remote_tmps/${JWM_SLURM_FILE}
SBATCH_OUT=$(sbatch ${sbatch_args} jwm_configs/${JWM_MODE}/remote_tmps/${JWM_SLURM_FILE}) || {
    return 1 2>/dev/null
    exit 1
}

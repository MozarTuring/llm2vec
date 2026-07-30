set -e

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
export RUN_DIR_HOME="/home/jinma"
export RUN_PROJ="llm2vec_jingwei"

eval "$(${RUN_DIR_HOME}/miniconda3/bin/conda shell.bash hook)"

if [ -n ${JWM_ENVS} ]; then
    if [ ! -d ${JWM_ENVS} ]; then
        conda create -p ${JWM_ENVS} python=3.11 -y
    fi
    conda activate ${JWM_ENVS}
    which python
    which pip
    if [ ! -d ${RUN_DIR_HOME}/jwmcondaenv/shared_cuda ]; then
        conda create -y -p ${RUN_DIR_HOME}/jwmcondaenv/shared_cuda -c nvidia cuda-toolkit
    fi
    export CUDA_HOME=${RUN_DIR_HOME}/jwmcondaenv/shared_cuda
    export PATH=${CUDA_HOME}/bin:${PATH}
    export CPATH=${CUDA_HOME}/targets/x86_64-linux/include:${CPATH}
    export LD_LIBRARY_PATH=${CUDA_HOME}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH}
fi


export CUDA_VISIBLE_DEVICES=0
export JWM_NOTEBOOK=1
export JWM_RUN_COMMAND="python experiments/run_layerwise_finetune.py train_configs/layerwise/MetaLlama3.1-mntp-layerwise.json"
export JWM_SERVER_NAME=greatrawr
export JWM_CONDAENV="${RUN_DIR_HOME}/conda_envs/llm2vec"
export JWM_GPU_NUM=
export JWM_NODES_NUM=
export JWM_RUN_TIME=
export JWM_build_flashattn=

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
#
# python experiments/download_model.py \
#     --model_name_or_path meta-llama/Meta-Llama-3.1-8B \
#     --dataset_name Tevatron/msmarco-passage-corpus

echo ${PWD}
JWM_RUN_COMMAND="${JWM_RUN_COMMAND_PRE} ${JWM_RUN_COMMAND}"

echo "JWM_RUN_COMMAND, 
${JWM_RUN_COMMAND}
"
kill $(pgrep -f "port=18889") || echo "18889 port free"
sleep 5

if [[ ${JWM_NOTEBOOK} == 1 ]]; then
    JWM_RUN_COMMAND="jupyter labextension disable '@jupyterlab/apputils-extension:announcements' && CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES}' jupyter lab --MappingKernelManager.cull_idle_timeout=3600 --MappingKernelManager.cull_interval=360 --MappingKernelManager.cull_connected=True --ip=0.0.0.0 --port=18889 --no-browser --allow-root --NotebookApp.token=''"
fi
nohup bash -c "CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES}' ${JWM_RUN_COMMAND}" > jwmlogs/${JWM_RUN_START_TIME}/job_out.log 2>&1 &

export JWM_JOB_ID=$!

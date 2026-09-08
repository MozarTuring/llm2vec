set -e 
export JWM_PYTHON="3.10"
export JWM_SERVER_NAME=greatrawr
export JWM_build_flashattn=
export JWM_NOTEBOOK=
export CUDA_VISIBLE_DEVICES=1
export JWM_RUN_COMMAND="python experiments/hard_negatives.py --top-k 1000 --num-top 50 --num-random 50 --output msmarco_hard_negatives_v2.json --query-batch-size 8192"
eval "$(${RUN_DIR_HOME}/miniconda3/bin/conda shell.bash hook)"

if [ -n ${JWM_PYTHON} ]; then
    if [ -z ${JWM_CONDAENV} ]; then
        JWM_CONDAENV=${RUN_DIR_HOME}/jwmcondaenv/${RUN_PROJ}
    fi
    echo "condaenv path ${JWM_CONDAENV}"
    if [ ! -d ${JWM_CONDAENV} ]; then
        conda create -p ${JWM_CONDAENV} python=${JWM_PYTHON} -y
    fi
    conda activate ${JWM_CONDAENV}
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
eval "$(${RUN_DIR_HOME}/miniconda3/bin/conda shell.bash hook)"

if [ -n ${JWM_PYTHON} ]; then
    if [ -z ${JWM_CONDAENV} ]; then
        JWM_CONDAENV=${RUN_DIR_HOME}/jwmcondaenv/${RUN_PROJ}
    fi
    echo "condaenv path ${JWM_CONDAENV}"
    if [ ! -d ${JWM_CONDAENV} ]; then
        conda create -p ${JWM_CONDAENV} python=${JWM_PYTHON} -y
    fi
    conda activate ${JWM_CONDAENV}
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

# hf download "naver/trecdl22-crossencoder-debertav3" \
#   --include "*" \
#   --local-dir "${JWM_DATA_DIR}/hf_models/naver/trecdl22-crossencoder-debertav3"

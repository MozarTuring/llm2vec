JWM_SERVER_NAME=berzeliusampere
export JWM_MODULES="Miniforge3 buildenv-gcccuda/12.4.1-gcc13.3.0"
export JWM_CONDAENV="/proj/berzelius-aiics-real/users/x_jinma/conda_envs/llm2vec"
export JWM_GPU_NUM=1
export JWM_NODES_NUM=1
export JWM_RUN_TIME="0-10:00:00"
export JWM_build_flashattn=
export JWM_SLURM_RUN_COMMAND="python experiments/run_word_task.py"
export JWM_SLURM_RUN_ARGS="train_configs/word-task/ShearedLlama-bi-mntp.json"
if [[ ${JWM_SLURM_RUN_ARGS} == "train_configs/mntp/MetaLlama3.json" ]]; then

    export JWM_SLURM_NODES="--nodelist=node[061-064,065,066-093]"
fi
if [[ -n ${JWM_build_flashattn} ]]; then
    export CPUS_PER_TASK=32
    export MEM_PER_TASK="256G"
else
    export CPUS_PER_TASK=$((8 * JWM_GPU_NUM))
    export MEM_PER_TASK="$((24 * JWM_GPU_NUM))G"
fi
export JWM_SLURM_FILE=slurm.sh

module --force purge
module load ${JWM_MODULES}

if [ ! -d ${JWM_CONDAENV} ]; then

    conda create -p ${JWM_CONDAENV} python=3.10 -y

fi
conda activate ${JWM_CONDAENV}
which python
echo $PWD

# pip install -e .
# pip install torch --force-reinstall --index-url https://download.pytorch.org/whl/cu124
# pip install ninja
# pip uninstall -y flash-attn 2>/dev/null
mkdir -p ${JWM_CONDAENV}/flash_attn_src
python -c "
import json, urllib.request, os
dest = os.environ['JWM_CONDAENV'] + '/flash_attn_src/flash_attn-2.8.3.post1.tar.gz'
if not os.path.exists(dest):
    data = json.loads(urllib.request.urlopen('https://pypi.org/pypi/flash-attn/2.8.3.post1/json').read())
    url = [u['url'] for u in data['urls'] if u['packagetype'] == 'sdist'][0]
    print(f'Downloading {url}')
    urllib.request.urlretrieve(url, dest)
    print('Done')
else:
    print('Source tarball already exists')
"

# pip install peft==0.12.0

# python experiments/download_model.py \
#     --model_name_or_path meta-llama/Meta-Llama-3.1-8B \
#     --dataset_name Tevatron/msmarco-passage-corpus

set -e

require_env() {
for var in "$@"; do
    if [ -z "${!var}" ]; then
        echo "Error: $var is not set" >&2
        exit 1
    fi
done
}
# change the following based on your running preference
export RUN_DIR_HOME="/home/x_jinma"
export RUN_PROJ="llm2vec_jingwei"

JWM_SERVER_NAME=berzeliusampere
export JWM_MODULES="Miniforge3 buildenv-gcccuda/12.4.1-gcc13.3.0"
export JWM_CONDAENV="/proj/berzelius-aiics-real/users/x_jinma/conda_envs/llm2vec"
export JWM_GPU_NUM=1
export JWM_NODES_NUM=1
export JWM_RUN_TIME="0-10:00:00"
export JWM_build_flashattn=
export JWM_SLURM_RUN_COMMAND="python experiments/test_word_task.py"
export JWM_SLURM_RUN_ARGS=" --config_file test_configs/word-task/ShearedLlama-bi-mntp.json"
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
# pip install datasets==3.6.0
pip install seqeval
# python experiments/download_model.py \
#     --model_name_or_path meta-llama/Meta-Llama-3.1-8B \
#     --dataset_name Tevatron/msmarco-passage-corpus

require_env JWM_SLURM_FILE JWM_RUN_TIME JWM_NODES_NUM
cat ${RUN_DIR_HOME}/project_remote_jwm/common_tools_jingwei/slurm_header.sh ${JWM_SLURM_FILE} > jwm_configs/${_mode}/remote_tmps/${JWM_SLURM_FILE}

sbatch_args="--time=${JWM_RUN_TIME} --nodes=${JWM_NODES_NUM} --output=jwmlogs/${JWM_RUN_START_TIME}/job-%j.out --error=jwmlogs/${JWM_RUN_START_TIME}/job-%j.out ${JWM_SLURM_NODES}"
sbatch_args="${sbatch_args} --gpus=${JWM_GPU_NUM} --cpus-per-task=${CPUS_PER_TASK} --mem=${MEM_PER_TASK} --signal=TERM@90 -A berzelius-2026-50 --partition=berzelius"

echo ${sbatch_args} jwm_configs/${_mode}/remote_tmps/${JWM_SLURM_FILE}
SBATCH_OUT=$(sbatch ${sbatch_args} jwm_configs/${_mode}/remote_tmps/${JWM_SLURM_FILE}) || {
    return 1 2>/dev/null
    exit 1
}

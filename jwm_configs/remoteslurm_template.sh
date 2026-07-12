JWM_SERVER_NAME=
export JWM_MODULES="Miniforge3 buildenv-gcccuda/12.4.1-gcc13.3.0"
export JWM_CONDAENV="/proj/berzelius-aiics-real/users/x_jinma/conda_envs/llm2vec"
export JWM_GPU_NUM=
export JWM_NODES_NUM=
export JWM_RUN_TIME=
export CPUS_PER_TASK=$((8 * JWM_GPU_NUM))
export MEM_PER_TASK="$((24 * JWM_GPU_NUM))G"
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

python experiments/download_model.py \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B \
    --dataset_name Tevatron/msmarco-passage-corpus

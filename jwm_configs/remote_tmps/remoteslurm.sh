JWM_SERVER_NAME=berzeliusampere
export JWM_MODULES="Miniforge3 buildenv-gcccuda/12.4.1-gcc13.3.0"
export JWM_CONDAENV="/proj/berzelius-aiics-real/users/x_jinma/conda_envs/llm2vec"
export JWM_GPU_NUM=1
export JWM_NODES_NUM=1
export JWM_RUN_TIME="0-10:00:00"
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

pip install torch --index-url https://download.pytorch.org/whl/cu124
pip uninstall -y flash-attn 2>/dev/null
pip install flash-attn --no-build-isolation

python experiments/download_model.py \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B \
    --dataset_name Tevatron/msmarco-passage-corpus

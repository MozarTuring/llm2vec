export JWM_MODULES="Miniforge3 buildenv-gcccuda/12.4.1-gcc13.3.0"
export JWM_CONDAENV="${RUN_DIR_HOME}/conda_envs/llm2vec"
if [[ ${JWM_SLURM_RUN_ARGS} == *"MetaLlama3"* ]]; then

    export JWM_SLURM_NODES="--nodelist=node[061-064,065,066-093]"
fi
module --force purge
module load ${JWM_MODULES}

if [ ! -d ${JWM_CONDAENV} ]; then

    conda create -p ${JWM_CONDAENV} python=3.10 -y

fi
conda activate ${JWM_CONDAENV}
which python
echo $PWD



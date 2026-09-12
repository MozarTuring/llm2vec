eval "$(${RUN_DIR_HOME}/miniconda3/bin/conda shell.bash hook)"
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
if [ ! -d ${RUN_DIR_HOME}/jwmcondaenv/shared_cuda ]; then
    conda create -y -p ${RUN_DIR_HOME}/jwmcondaenv/shared_cuda -c nvidia cuda-toolkit
fi
export CUDA_HOME=${RUN_DIR_HOME}/jwmcondaenv/shared_cuda
export PATH=${CUDA_HOME}/bin:${PATH}
export CPATH=${CUDA_HOME}/targets/x86_64-linux/include:${CPATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH}

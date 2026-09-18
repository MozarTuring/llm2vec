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

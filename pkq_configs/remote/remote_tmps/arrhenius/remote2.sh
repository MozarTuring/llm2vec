if [[ -z ${PKQ_MODULES} ]]; then
export PKQ_MODULES="GPU/Miniforge/26.3.2-2-eb"
fi
export PKQ_ARCH="aarch64"

if [ -z ${PKQ_CONDAENV} ]; then
    export PKQ_CONDAENV=${RUN_DIR_HOME}/pkqcondaenv/${RUN_PROJ}
    export PKQ_WHEELS=${RUN_DIR_HOME}/pkqwheels/${RUN_PROJ}
fi
echo "condaenv path ${PKQ_CONDAENV}"

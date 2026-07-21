#!/bin/bash

(while true; do echo ""; echo "CPU Usage: $(vmstat 1 2 | tail -1 | awk '{print 100 - $15}')% | Total CPUs: $(nproc)"; nvidia-smi; echo ""; sleep 300; done) > jwmlogs/${JWM_RUN_START_TIME}/resource_usage.log 2>&1 &


module --force purge
if [[ -n "${JWM_MODULES}" ]];then
    echo ${JWM_MODULES}
    module load ${JWM_MODULES}
fi

if [[ -n "${JWM_CONDAENV}" ]];then
echo ${JWM_CONDAENV}
conda activate ${JWM_CONDAENV}
fi


which python

if [[ -n ${JWM_build_flashattn} ]]; then
    MAX_JOBS=${CPUS_PER_TASK} FLASH_ATTENTION_FORCE_BUILD=TRUE pip install ${JWM_CONDAENV}/flash_attn_src/flash_attn*.tar.gz --no-build-isolation --no-cache-dir
    echo "flash attn build done"
    exit
fi

export LD_LIBRARY_PATH=${LIBRARY_PATH}:${LD_LIBRARY_PATH:-}

${JWM_SLURM_RUN_COMMAND} ${JWM_SLURM_RUN_ARGS}

rm /home/x_jinma/project_remote_jwm/llm2vec_jingwei/20260721_081841.jwm


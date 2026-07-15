#!/bin/bash

(while true; do echo "CPU Usage: $(vmstat 1 2 | tail -1 | awk '{print 100 - $15}')% | Total CPUs: $(nproc)"; nvidia-smi; sleep 300; done) &



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
python experiments/run_mntp.py train_configs/mntp/MetaLlama3.1-msmarco.json

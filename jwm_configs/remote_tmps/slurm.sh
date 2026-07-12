#!/bin/bash

(while true; do echo "nvidia-smi"; nvidia-smi; sleep 300; done) &

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


python experiments/run_mntp.py train_configs/mntp/MetaLlama3.1-msmarco.json

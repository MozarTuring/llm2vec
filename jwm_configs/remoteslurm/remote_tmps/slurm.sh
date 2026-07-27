#!/bin/bash

early_warning() {
    echo "2 minutes left — saving checkpoint..."
    # save_checkpoint
    # optionally keep running, or exit gracefully
}

final_cleanup() {
    echo "Being killed — last-resort cleanup..."
    rm ${RUN_DIR_HOME}/project_remote_jwm/${RUN_PROJ}/${JWM_RUN_START_TIME}.jwm
}

trap early_warning SIGUSR1    # 120s before limit — your warning

trap final_cleanup SIGTERM    # 0s — SLURM is killing you


(while true; do
    echo ""
    echo "CPU Usage: $(vmstat 1 2 | tail -1 | awk '{print 100 - $15}')% | Total CPUs: $(nproc)"
    nvidia-smi
    echo ""
    sleep 300
done) >jwmlogs/${JWM_RUN_START_TIME}/resource_usage.log 2>&1 &

module --force purge
if [[ -n "${JWM_MODULES}" ]]; then
    echo ${JWM_MODULES}
    module load ${JWM_MODULES}
fi

if [[ -n "${JWM_CONDAENV}" ]]; then
    echo ${JWM_CONDAENV}
    conda activate ${JWM_CONDAENV}
fi
which python


if [[ -n ${JWM_build_flashattn} ]]; then
    MAX_JOBS=${CPUS_PER_TASK} FLASH_ATTENTION_FORCE_BUILD=TRUE pip install ${JWM_CONDAENV}/flash_attn_src/flash_attn*.tar.gz --no-build-isolation --no-cache-dir
    echo "flash attn build done"
    exit
fi




export LD_LIBRARY_PATH=/software/sse/manual/CUDA/12.4.1_550.54.15/lib64:/software/sse/manual/CUDA/12.4.1_550.54.15/extras/CUPTI/lib64:/software/sse/manual/ScaLAPACK/2.2.0/gcc-13.3.0/openmpi-5.0.3/openblas-0.3.27/lib:/software/sse/manual/OpenBLAS/0.3.27/gcc-13.3.0/sequential/lp64/lib:/software/sse/manual/FFTW/3.3.10/gcc-13.3.0/openmpi-5.0.3/lib:/software/sse/manual/OpenMPI/5.0.3/gcc-13.3.0/cuda-12.4.1/hpc1/lib:/software/sse/manual/GCC/13.3.0/lib64:

jupyter lab --MappingKernelManager.cull_idle_timeout=3600 --MappingKernelManager.cull_interval=360 --MappingKernelManager.cull_connected=True --ip=0.0.0.0 --port=18889 --no-browser --allow-root --NotebookApp.token=''  &
wait $!


#!/bin/bash

early_warning() {
    echo "2 minutes left — saving checkpoint..."
    # save_checkpoint
    # optionally keep running, or exit gracefully
}

final_cleanup() {
    echo "Being killed — last-resort cleanup..."
}

trap early_warning SIGUSR1 # 120s before limit — your warning

trap final_cleanup SIGTERM # 0s — SLURM is killing you

module --force purge
module load ${PKQ_MODULES}

for _eroot in "$EBROOTCUDA/lib64" "$EBROOTCUDNN/lib" "$EBROOTCUSPARSELT/lib"; do
    [[ -d "$_eroot" ]] && export LD_LIBRARY_PATH="${_eroot}:${LD_LIBRARY_PATH:-}"
done


conda activate ${PKQ_CONDAENV}${PKQ_ARCH}

which python
python -m  pip list >pkq_configs/packages.txt

bash ${RUN_DIR_HOME}/project_remote_pkq/common_tools_pikaq/resource_usage.sh >pkqlogs/${PKQ_RUN_START_TIME}/resource_usage.log &

echo "TORCH_CUDA_ARCH_LIST ${TORCH_CUDA_ARCH_LIST}"

if [[ -n ${JWM_build_flashattn} ]]; then
    MAX_JOBS=${CPUS_PER_TASK} FLASH_ATTENTION_FORCE_BUILD=TRUE pip install ${PKQ_CONDAENV}/flash_attn_src/flash_attn*.tar.gz --no-build-isolation --no-cache-dir
    echo "flash attn build done"
    exit
fi
if [[ "${PKQ_ARCH}" == "aarch64" ]]; then
    pip install --no-index --find-links "${_wheel_dir}" \
        torch \
        ninja \
        "datasets==3.6.0" \
        seqeval \
        jupyterlab \
        sentence_transformers \
        sentencepiece \
        protobuf \
        "peft==0.12.0" \
        mteb \
        ir_datasets \
        huggingface_hub \
        ijson \
        tomli \
        overrides
    pip install --no-deps -e .
fi
# Copy LIBRARY_PATH to LD_LIBRARY_PATH, but strip stubs dirs —
# CUDA module's stubs/lib64 has fake libnvidia-ml.so/libcuda.so
# that shadow the real driver and break GPU init.
_lib_path_no_stubs=$(echo "${LIBRARY_PATH:-}" | tr ':' '\n' | grep -v '/stubs/' | paste -sd ':')
export LD_LIBRARY_PATH=${_lib_path_no_stubs:+${_lib_path_no_stubs}:}${LD_LIBRARY_PATH:-}
echo "PKQ_RUN_COMMAND, ${PKQ_RUN_COMMAND}"
srun --ntasks=1 ${PKQ_RUN_COMMAND} &

wait $!

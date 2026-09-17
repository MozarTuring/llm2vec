## Download aarch64 wheels on login node (x86_64) for offline install on compute nodes

# pip install -e .
# pip install torch --force-reinstall --index-url https://download.pytorch.org/whl/cu128
# pip install ninja
# pip install datasets==3.6.0
# pip install seqeval
# pip install jupyterlab
# pip install sentence_transformers
# pip install sentencepiece
# pip install protobuf
#
# pip install peft==0.12.0

# pip install mteb
# pip install ir_datasets
# pip install -q huggingface_hub
# pip install ijson

if [[ "${JWM_ARCH}" == "aarch64" ]]; then
    _wheel_dir=${JWM_WHEELS}${JWM_ARCH}
    PLATFORM="manylinux2014_aarch64"

    mkdir -p "${_wheel_dir}"

    pip download \
        --platform "${PLATFORM}" \
        --python-version "${JWM_PYTHON}" \
        --only-binary=:all: \
        -d "${_wheel_dir}" \
        torch --index-url https://download.pytorch.org/whl/cu128

    pip download \
        --platform "${PLATFORM}" \
        --python-version "${JWM_PYTHON}" \
        --only-binary=:all: \
        -d "${_wheel_dir}" \
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
        ijson

fi

# python experiments/download_model.py \
#     --model_name_or_path meta-llama/Meta-Llama-3.1-8B \
#     --dataset_name Tevatron/msmarco-passage-corpus

# hf download "OpenMOSS-Team/Llama3_1-8B-Base-LXR-8x" \
#   --include "Llama3_1-8B-Base-L26R-8x/*" \
#   --local-dir "${JWM_DATA_DIR}"

# hf download "naver/trecdl22-crossencoder-debertav3" \
#   --include "*" \
#   --local-dir "${JWM_DATA_DIR}/hf_models/naver/trecdl22-crossencoder-debertav3"

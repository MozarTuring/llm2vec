if [[ "${JWM_ARCH}" == "aarch64" ]]; then
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
        tomli
    pip install --no-deps -e .
fi

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
#
# pip install mteb
# pip install ir_datasets
# pip install -q huggingface_hub
# pip install ijson
#
# pip install --force-reinstall transformers==4.44.2


# hf download Tevatron/msmarco-passage-corpus --repo-type dataset

# hf download "meta-llama/Meta-Llama-3.1-8B" --exclude "*.pth"
# hf download "OpenMOSS-Team/Llama3_1-8B-Base-LXR-8x" \
#   --include "Llama3_1-8B-Base-L26R-8x/*"
# hf download "OpenMOSS-Team/Llama3_1-8B-Base-LXR-8x" \
#   --include "Llama3_1-8B-Base-L0R-8x/*"
# hf download "naver/trecdl22-crossencoder-debertav3"

# hf download OpenMOSS-Team/Llama3_1-8B-Base-LXR-32x --include "Llama3_1-8B-Base-L26R-32x/*"
hf download naver/splade-v3

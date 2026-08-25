# pip install -e .
# pip install torch --force-reinstall --index-url https://download.pytorch.org/whl/cu128
# pip install ninja
# pip uninstall -y flash-attn 2>/dev/null
# mkdir -p ${JWM_CONDAENV}/flash_attn_src
# python -c "
# import json, urllib.request, os
# dest = os.environ['JWM_CONDAENV'] + '/flash_attn_src/flash_attn-2.8.3.post1.tar.gz'
# if not os.path.exists(dest):
#     data = json.loads(urllib.request.urlopen('https://pypi.org/pypi/flash-attn/2.8.3.post1/json').read())
#     url = [u['url'] for u in data['urls'] if u['packagetype'] == 'sdist'][0]
#     print(f'Downloading {url}')
#     urllib.request.urlretrieve(url, dest)
#     print('Done')
# else:
#     print('Source tarball already exists')
# "

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
#
# python experiments/download_model.py \
#     --model_name_or_path meta-llama/Meta-Llama-3.1-8B \
#     --dataset_name Tevatron/msmarco-passage-corpus


# hf download "OpenMOSS-Team/Llama3_1-8B-Base-LXR-8x" \
#   --include "Llama3_1-8B-Base-L26R-8x/*" \
#   --local-dir "${JWM_DATA_DIR}"

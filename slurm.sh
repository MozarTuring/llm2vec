
which python

FLASH_ATTENTION_FORCE_BUILD=TRUE pip install ${JWM_CONDAENV}/flash_attn_src/flash_attn*.tar.gz --no-build-isolation --no-cache-dir

python experiments/run_mntp.py train_configs/mntp/MetaLlama3.1-msmarco.json

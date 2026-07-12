
which python

# pip uninstall -y flash-attn 2>/dev/null
# pip install flash-attn --no-build-isolation

python experiments/run_mntp.py train_configs/mntp/MetaLlama3.1-msmarco.json

JWM_PYTHON="3.10"
JWM_SERVER_NAME=greatrawr
JWM_build_flashattn=
JWM_NOTEBOOK=
CUDA_VISIBLE_DEVICES=0


# JWM_RUN_COMMAND="python experiments/run_layerwise_finetune.py train_configs/layerwise/MetaLlama3.1-mntp-layerwise.json"

JWM_RUN_COMMAND="python experiments/mteb_eval_layerwise.py --model_name_or_path meta-llama/Meta-Llama-3.1-8B --peft_model_name_or_path output/mntp/Meta-Llama-3.1-8B-msmarco --sae_weights_path /home/jinma/project_remote_jwm/remote_data/splare/Llama3_1-8B-Base-L0R-8x/checkpoints/final.safetensors --trained_checkpoint_path output/layerwise/Meta-Llama-3.1-8B-msmarco-mntp-L0/checkpoint-623 --lora_layers 0 --task_name SciFact --output_dir results --query_top_k 40 --doc_top_k 400"

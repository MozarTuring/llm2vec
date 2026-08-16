set -e 
export JWM_NOTEBOOK=1
export CUDA_VISIBLE_DEVICES=0
export JWM_PYTHON="3.10"
export JWM_build_flashattn=
export JWM_SERVER_NAME=greatrawr
export JWM_RUN_COMMAND="python experiments/mteb_eval_layerwise.py --model_name_or_path meta-llama/Meta-Llama-3.1-8B --peft_model_name_or_path output/mntp/Meta-Llama-3.1-8B-msmarco --sae_weights_path /home/jinma/project_remote_jwm/remote_data/splare/Llama3_1-8B-Base-L0R-8x/checkpoints/final.safetensors --trained_checkpoint_path output/layerwise/Meta-Llama-3.1-8B-msmarco-mntp-L0/checkpoint-623 --lora_layers 0 --task_name SciFact --output_dir results --query_top_k 40 --doc_top_k 400"

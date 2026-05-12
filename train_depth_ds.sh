export CUDA_VISIBLE_DEVICES=$1

# Model Paths
base_model_path="/home/majunjie/code/weights/hf/DeepSeek-R1-Distill-Llama-8B"
ea_model_path="/home/majunjie/code/weights/eagle/EAGLE3-DeepSeek-R1-Distill-LLaMA-8B"
rl_token_model_path="/home/majunjie/code/Learning-to-Draft/checkpoints/deepseek/size/ppo_speculative_decoder_controller_rebuttal.zip" # Leave empty to use default 60 tokens
rl_checkpoint_path=""  # Leave empty to start from scratch

# Data and Save Directories
data_dir="./eagle/data"
dataset_train="humaneval"
save_path="./checkpoints/deepseek/depth/"

# RL Hyperparameters
total_timesteps=1000000
batch_size=64
n_steps=128
lr=3e-4

python3 -m rl.rl_depth \
    --base_model_path ${base_model_path} \
    --ea_model_path ${ea_model_path} \
    --rl_token_model_path "${rl_token_model_path}" \
    --rl_checkpoint_path "${rl_checkpoint_path}" \
    --data_dir ${data_dir} \
    --dataset_train ${dataset_train} \
    --save_path ${save_path} \
    --total_timesteps ${total_timesteps} \
    --batch_size ${batch_size} \
    --n_steps ${n_steps} \
    --lr ${lr} \
    --pi_arch 1024 \
    --vf_arch 1024 256
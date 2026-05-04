export CUDA_VISIBLE_DEVICES=$1

# Model Paths
base_model_path="/home/majunjie/code/weights/hf/Meta-Llama-3.1-8B-Instruct"
ea_model_path="/home/majunjie/code/weights/eagle/EAGLE3-LLaMA3.1-Instruct-8B"
rl_token_model_path="/home/majunjie/code/Learning-to-Draft/checkpoints/size_llama3_2.zip" # Leave empty to use default 60 tokens
rl_checkpoint_path="/home/majunjie/code/Learning-to-Draft/checkpoints/depth_llama3_1.zip"  # Leave empty to start from scratch

# Data and Save Directories
data_dir="./eagle/data"
dataset_train="humaneval"
save_path="./checkpoints/llama3_final"

# RL Hyperparameters
total_timesteps=100000
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
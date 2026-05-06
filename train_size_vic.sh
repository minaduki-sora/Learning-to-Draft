export CUDA_VISIBLE_DEVICES=$1

# Model Paths
base_model_path="/home/majunjie/code/weights/hf/vicuna-13b-v1.3"
ea_model_path="/home/majunjie/code/weights/eagle/EAGLE3-Vicuna1.3-13B"
depth_model="" # Leave empty to use default 60 tokens
rl_checkpoint_path=""  # Leave empty to start from scratch

# Data and Save Directories
data_dir="./eagle/data"
dataset_train="humaneval"
save_path="./checkpoints/vicuna/size"

# RL Hyperparameters
total_timesteps=100000
batch_size=64
n_steps=128
lr=3e-4

python3 -m rl.rl_total \
    --base_model_path ${base_model_path} \
    --ea_model_path ${ea_model_path} \
    --depth_model "${depth_model}" \
    --max_memory_gpu "23GiB" --max_memory_cpu "60GiB"\
    --rl_checkpoint_path "${rl_checkpoint_path}" \
    --data_dir ${data_dir} \
    --dataset_train ${dataset_train} \
    --save_path ${save_path} \
    --total_timesteps ${total_timesteps} \
    --batch_size ${batch_size} \
    --n_steps ${n_steps} \
    --lr ${lr} \
    --pi_arch 1024 256 \
    --vf_arch 1024 256
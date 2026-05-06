DATASETS=(alpaca mt_bench mbpp gsm8k)
Depth_Policy="/home/majunjie/code/Learning-to-Draft/checkpoints/llama3_sd/ppo_speculative_decoder_controller_v1_single_action.zip"
Size_Policy="/home/majunjie/code/Learning-to-Draft/checkpoints/llama3_final/size_llama3_1.zip"


for data in "${DATASETS[@]}"; do
    echo "  -> 正在运行数据集: $data"
    
    CUDA_VISIBLE_DEVICES=1 python -m eagle.evaluation.gen_ea_answer_llama3chat \
        --bench-name "$data" \
        --depth 8 \
        --temperature 1 \
        --num-choices 1 \
        --total-token 60 \
        --use_dyn_depth \
        --use_dyn_token \
        --token_model ${Size_Policy} \
        --depth_model ${Depth_Policy}
done

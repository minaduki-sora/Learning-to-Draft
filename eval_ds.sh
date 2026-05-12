DATASETS=(alpaca mt_bench mbpp gsm8k)
Depth_Policy="/home/majunjie/code/Learning-to-Draft/checkpoints/deepseek/depth/ppo_speculative_decoder_controller_v1_single_action.zip"
Size_Policy="/home/majunjie/code/Learning-to-Draft/checkpoints/deepseek/size/ppo_speculative_decoder_controller_rebuttal.zip"


for data in "${DATASETS[@]}"; do
    echo "  -> 正在运行数据集: $data"
    
    CUDA_VISIBLE_DEVICES=2 python -m eagle.evaluation.gen_ea_answer_ds \
        --bench-name "$data" \
        --depth 8 \
        --temperature 1 \
        --num-choices 5 \
        --total-token 60 \
        --use_dyn_depth \
        --use_dyn_token \
        --token_model ${Size_Policy} \
        --depth_model ${Depth_Policy} \
        --num-gpus-total 2 \
        --num-gpus-per-model 2
done

export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

CUDA_VISIBLE_DEVICES=0,1 python3 -m accelerate.commands.launch \
    --num_processes=2 \
    --main_process_port 29881 \
    -m lmms_eval \
    --model llava \
    --model_args pretrained="liuhaotian/llava-v1.6-vicuna-7b" \
    --tasks pope_pop \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_v1.6_pope \
    --output_path ./logs/ \
    --generation_type recursion \
    --fix_grid 2x2 \
    --attention_thresholding_type attn_topk \
    --attention_threshold 1.0 \
    --positional_embedding_type bilinear_interpolation \
    --stages "-2" "-1" "0" "1" \
    --verbosity DEBUG \
    --contrastive_alphas "0.8" "0.8" "0.8"

# CUDA_VISIBLE_DEVICES=0,1 python3 -m accelerate.commands.launch \
#     --num_processes=2 \
#     --main_process_port 29881 \
#     -m lmms_eval \
#     --model llava \
#     --model_args pretrained="liuhaotian/llava-v1.6-vicuna-7b" \
#     --tasks vqav2_val_lite \
#     --batch_size 1 \
#     --log_samples \
#     --log_samples_suffix llava_v1.6_pope \
#     --output_path ./logs/ \
#     --generation_type recursion \
#     --fix_grid 2x2 \
#     --attention_thresholding_type attn_topk \
#     --attention_threshold 0.5 \
#     --positional_embedding_type bilinear_interpolation \
#     --stages "-2" "-1" "0" "1" \
#     --verbosity DEBUG \
#     --contrastive_alphas "0.1" "0.2" "0.1"

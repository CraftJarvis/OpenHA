#! /bin/zsh

cuda_visible_devices=0,1,2,3

CUDA_VISIBLE_DEVICES=$cuda_visible_devices python examples/online/rollout_coa_openha.py \
    --output_mode "greedy" \
    --model_path /share_data/limuyao/checkpoints/train/mc-motion-coa-qwen2-vl-7b-250809-A800-c32-e1-b8-a1/ \
    --model_id mc-motion-coa-qwen2-vl-7b-250809 \
    --sam_path "/share/personal/craftjarvis/models/sam2" \
    --grounding_policy_path "CraftJarvis/MineStudio_ROCKET-1.12w_EMA" \
    --motion_policy_path "/share/personal/craftjarvis/models/motion_ckpt-epoch=4-step=466233.ckpt" \
    --model_ip "localhost" \
    --model_ports "11000,11001" \
    --vlm_client_mode "vllm" \
    --maximum_history_length 2 \
    --record_path "/share_data/limuyao/evaluate/" \
    --num_steps 300 \
    --temperature 0.7 \
    --top_p 0.99 \
    --top_k -1

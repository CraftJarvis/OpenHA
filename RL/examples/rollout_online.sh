python examples/rollout_openha.py --output_mode text_action  \
    --vlm_client_mode online \
    --system_message_tag text_action \
    --model_ips localhost \
    --model_ports 11000 \
    --model_id CrossAgent-qwen2vl-7b \
    --record_path "output" \
    --max_steps_num 200 \
    --maximum_history_length 3 \
    --num_rollouts 1 # 1 for debug
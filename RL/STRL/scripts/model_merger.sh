python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "/share/hkc/checkpoints/rl-craft/mc-mix-coa/mc-mix-coa-qwen2-vl-7b-250906/global_step_120/actor" \
    --target_dir "/share/hkc/checkpoints/ssrl/mc-mix-coa-qwen2-vl-7b-250906/global_step_120"


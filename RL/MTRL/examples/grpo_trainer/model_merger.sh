python scripts/model_merger.py merge \
    --backend fsdp \
    --local_dir "/share/hkc/checkpoints/asyncRL/mix_coa/global_step_90/actor" \
    --target_dir "/share/hkc/checkpoints/25cvpr_msrl/asyncRL/mix_coa_rl_global_step_90"

python scripts/model_merger.py merge \
    --backend fsdp \
    --local_dir "/share/hkc/checkpoints/asyncRL/mix_coa_251103/global_step_140/actor" \
    --target_dir "/share/hkc/checkpoints/25cvpr_msrl/asyncRL/mix_coa_251103_rl_global_step_140"
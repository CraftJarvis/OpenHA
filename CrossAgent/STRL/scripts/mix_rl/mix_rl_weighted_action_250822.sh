set -x
ENGINE=${1:-vllm}

project_name='mc-mix-action_balanced'
experiment_name='mc-mix-action_balanced-coa-qwen2-vl-7b-250822_1900'

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    custom_reward_function.path=OpenHA/rl/custom_reward_funcs/mix_vla.py \
    data.train_files=OpenHA/rl/datasets/mc-mix-action_balanced/train.parquet \
    data.val_files=OpenHA/rl/datasets/mc-mix-action_balanced/valid.parquet \
    data.train_batch_size=1024 \
    data.max_prompt_length=4096 \
    data.max_response_length=4096\
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.image_key=images \
    actor_rollout_ref.model.path=/share/hkc/checkpoints/mc-mix-coa-qwen2-vl-7b-250816-A800-c8-e1-b1-a2/checkpoint-400  \
    data.tokenizer=/share/hkc/checkpoints/mc-mix-coa-qwen2-vl-7b-250816-A800-c8-e1-b1-a2/checkpoint-400 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','swanlab'] \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.default_local_dir=/share/hkc/checkpoints/rl-craft/${project_name}/${experiment_name} \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=15 $@
    #trainer.rollout_data_dir=OpenHA/rl/rollout_data_dir \
    
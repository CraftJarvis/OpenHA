set -x
ENGINE=${1:-vllm}
export VLLM_ATTENTION_BACKEND=XFORMERS

train_data_size=1
val_data_size=1
group_size=1

project_name='verl_agent_minecraft_dynamicsampling'
experiment_name='grpo'
# We only use data preparation to indicate the modality and the data size.
python3 -m examples.data_preprocess.prepare \
    --mode 'text' \
    --train_data_size $train_data_size \
    --val_data_size $val_data_size


#/share_data/limuyao/checkpoints/train/mc-coa-craft-qwen2-vl-7b-250725-A800-c32-e1-b8-a1/checkpoint-2998 \
RAY_DEBUG=legacy python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.filter_groups.enable=False \
    algorithm.dynamic_rollouts=True \
    algorithm.filter_groups.max_num_gen_batches=4 \
    data.train_files=$HOME/data/verl-agent/text/train.parquet \
    data.val_files=$HOME/data/verl-agent/text/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=2048 \
    data.max_response_length=128 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=/share_data/limuyao/checkpoints/train/mc-openha-state2-qwen2-vl-7b-250830-A800-e1-b4-a1/checkpoints/global_step_300/hf_ckpt\
    actor_rollout_ref.actor.optim.lr=5e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    env.env_name=minecraft \
    env.seed=0 \
    env.task_path=OpenHA/rl/data_processor/utils/task_list.json \
    env.record_path=MC-verl-agent/examples/grpo_trainer/output/videos/${project_name}/${experiment_name} \
    env.maximum_history_length=5 \
    env.max_steps=192 \
    env.rollout.n=$group_size \
    trainer.critic_warmup=0 \
    trainer.logger=['console','swanlab'] \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.default_local_dir=checkpoints/${project_name}/${experiment_name} \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=150 \
    trainer.val_before_train=False $@

echo "========== Checking dmesg for killed processes =========="
dmesg -T | grep -i kill
set -x
ENGINE=${1:-vllm}
export VLLM_ATTENTION_BACKEND=XFORMERS

nnodes=1
train_data_size=8
val_data_size=$nnodes
group_size=4

project_name='verl_agent_minecraft_dynamicsampling'
experiment_name='motion_coa_0929'
# We only use data preparation to indicate the modality and the data size.
python3 -m examples.data_preprocess.prepare \
    --mode 'text' \
    --train_data_size $train_data_size \
    --val_data_size $nnodes

#????? 
# export NCCL_P2P_DISABLE=0
# export NCCL_P2P_LEVEL=NVL

# # 2) 开启更友好的错误上报（早点 fail，别拖 30 分钟）
# export NCCL_ASYNC_ERROR_HANDLING=1
# export TORCH_NCCL_BLOCKING_WAIT=1

# # 3) 不要禁用共享内存；增大 /dev/shm（容器内用 --shm-size=64g 或更大）
# export NCCL_SHM_DISABLE=0

# # 4) 单机时避免走 IB/NET（有时 RDMA 配置不一致会搅局）
# #   若你是“严格单机训练”，可以先强制 Socket，排除 IB 干扰：
# export NCCL_IB_DISABLE=1 这个比较重要？

#！！！export NCCL_P2P_DISABLE=1否则会nccl watchdog timeout, 但这会慢一倍！
#/share_data/limuyao/checkpoints/train/mc-coa-craft-qwen2-vl-7b-250725-A800-c32-e1-b8-a1/checkpoint-2998 \
#gpu memory_utilization=0.4
#env.rollout_path=NoneMC-verl-agent/examples/grpo_trainer/output/videos/${project_name}/${experiment_name} \

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.filter_groups.enable=True \
    algorithm.dynamic_rollouts=True \
    algorithm.filter_groups.max_num_gen_batches=1 \
    data.train_files=$HOME/data/verl-agent/text/train.parquet \
    data.val_files=$HOME/data/verl-agent/text/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=3072 \
    data.max_response_length=256 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=/share/hkc/checkpoints/msrl_coldstart/mc-openha-state2-qwen2-vl-7b-motion_250929-A800-e1-b4-a1-s19456/checkpoints/global_step_200/hf_ckpt \
    actor_rollout_ref.actor.optim.lr=5e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.limit_images=4 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    env.env_name=minecraft \
    env.seed=0 \
    env.task_path=OpenHA/rl/data_processor/utils/task_list.json \
    env.maximum_history_length=3 \
    env.max_steps=200 \
    env.rollout.n=$group_size \
    trainer.critic_warmup=0 \
    trainer.logger=['console','swanlab'] \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.default_local_dir=/share/hkc/checkpoints/${project_name}/${experiment_name} \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=$nnodes \
    trainer.save_freq=5 \
    trainer.test_freq=140 \
    trainer.total_epochs=150 \
    trainer.val_before_train=False $@

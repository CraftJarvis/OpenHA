#!/usr/bin/env bash
set -euo pipefail
set -x

########################
# 默认配置
########################


epoch=1
card_type="A800"
max_seq_length=19456 #12288
dataset_paths="OpenHA/rl/datasets/sft/msrl_coldstart/20250923/mc-msrl_coldstart-text_250923-train-withsystem.jsonl" #"/DATA/lmy/datasets/mc-mix_vla-0827-train.jsonl" 
real_dataset_len=4000 # 900000/2
base_model_path="/share_data/craftjarvis/models/mc-openha-stage2-qwen2-vl-7b-2509" 
version="mc-openha-state2-qwen2-vl-7b-text_250929" #stage2

cuda_visible_devices="0,1,2,3,4,5,6,7"
card_number=8          # 每机GPU数 = 每机进程数
node_number=1         # 总机器数

ulysses_parallel_size=4 
micro_batch_size=4
gradient_accumulation_steps=1

# 分布式相关（默认值，可被 env 或 CLI 覆盖）
NNODES=${NNODES:-$node_number}
NPROC_PER_NODE=${NPROC_PER_NODE:-$card_number}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-"0.0.0.0"}
MASTER_PORT=${MASTER_PORT:-24003}

########################
# 解析命令行参数
########################
# 支持：
#   --node-rank <int>
#   --master-addr <ip/host>
# 其余参数原样透传给 torchrun 之后的 Python 程序
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --node-rank)
      NODE_RANK="$2"; shift 2;;
    --master-addr)
      MASTER_ADDR="$2"; shift 2;;
    --master-port)
      MASTER_PORT="$2"; shift 2;;
    *)
      EXTRA_ARGS+=("$1"); shift;;
  esac
done

total_workers=$((NPROC_PER_NODE * NNODES / ulysses_parallel_size))
global_batch_size=$((total_workers * micro_batch_size * gradient_accumulation_steps))
max_steps=$(( real_dataset_len / global_batch_size ))

# 计算 warmup 步数
warmup_steps=$(( max_steps * 3 / 100 ))   # 相当于 max_steps * 0.03

# 下限 50
if (( warmup_steps < 50 )); then
  warmup_steps=50
fi

# 上限 200
if (( warmup_steps > 200 )); then
  warmup_steps=200
fi

# 不超过 max_steps - 1
if (( warmup_steps >= max_steps )); then
  warmup_steps=$(( max_steps - 1 ))
fi

# 换算成 ratio
lr_warmup_ratio=$(echo "scale=6; $warmup_steps / $max_steps" | bc -l)
echo "warmup_steps=$warmup_steps"
echo "lr_warmup_ratio=$lr_warmup_ratio"

WANDB_NAME="$version-$card_type-e$epoch-b$micro_batch_size-a$gradient_accumulation_steps-s$max_seq_length"
WANDB_PROJECT="VLA"

# 日志 & 追踪
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="$cuda_visible_devices"

export SWANLAB_API_KEY="u118Bhu4Uxl4PSj527zll"
export WANDB_API_KEY="9775ea57c312e2b1445afe756e7e68b72a1307b7"
export WANDB_PROJECT=$WANDB_PROJECT
export WANDB_NOTES="[25-05-2] vision language convs convs (instruction state action),with image augment"

########################
# 启动
########################
torchrun --nnodes="$NNODES" --nproc-per-node "$NPROC_PER_NODE" --node-rank "$NODE_RANK" \
  --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" \
  VeOmni/tasks/omni/train_qwen2_5_vl.py \
  VeOmni/configs/multimodal/qwen2_vl/qwen2_vl.yaml \
  --model.model_path "$base_model_path" \
  --data.train_path "$dataset_paths" \
  --data.chat_template qwen2_5vlwithfocal_onlylaststep \
  --data.train_size 100000000000000000000 \
  --train.max_steps $max_steps \
  --data.max_seq_len $max_seq_length \
  --data.source_name craftjarvis \
  --data.dataloader_type "native"  \
  --data.datasets_type "mapping" \
  --train.rmpad_with_pos_ids true \
  --train.seed 42 \
  --train.lr 8e-6 \
  --train.vit_lr 4e-6 \
  --train.lr_min 1e-6 \
  --train.lr_warmup_ratio $lr_warmup_ratio \
  --train.weight_decay 0.05 \
  --train.lr_decay_style cosine \
  --train.save_steps 100 \
  --train.output_dir "/share/hkc/checkpoints/msrl_coldstart/$WANDB_NAME" \
  --train.data_parallel_mode fsdp2 \
  --train.wandb_project "$WANDB_PROJECT" \
  --train.wandb_name "$WANDB_NAME" \
  --train.num_train_epochs "$epoch" \
  --train.dyn_bsz_buffer_size 200 \
  --train.ulysses_parallel_size $ulysses_parallel_size \
  --train.context_parallel_size 1 \
  --train.tensor_parallel_size 1 \
  --train.expert_parallel_size 1 \
  --train.pipeline_parallel_size 1 \
  --train.global_batch_size $global_batch_size \
  --train.micro_batch_size $micro_batch_size \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee log.txt

  #--train.load_checkpoint_path "/share_data/limuyao/checkpoints/train/mc-text_vla-craft-qwen2-vl-7b-250823-A800-e1-b4-a1/checkpoints/global_step_1600" \
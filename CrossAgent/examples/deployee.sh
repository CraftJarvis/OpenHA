# CrossAgent: CraftJarvis/CrossAgent-qwen2vl-7b
# TextHA_RL: CraftJarvis/TextHA-RL-qwen2vl-7b
# GroundingHA_RL: CraftJarvis/GroundingHA-RL-qwen2vl-7b
# MotionHA_RL: CraftJarvis/MotionHA-RL-qwen2vl-7b
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 vllm serve CraftJarvis/CrossAgent-qwen2vl-7b  \
    --served-model-name CrossAgent-qwen2vl-7b  \
    --port 11000 \
    --limit-mm-per-prompt image=5  \
    --trust-remote-code --gpu-memory-utilization 0.90  \
    --pipeline-parallel-size 1  \
    --tensor-parallel-size 1  \
    --max-num-seqs 16 \
    --max-logprobs 20 \
    --max-model-len 32768
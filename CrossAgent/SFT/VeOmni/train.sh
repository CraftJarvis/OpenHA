#!/bin/bash

set -x

export TOKENIZERS_PARALLELISM=false
cuda_visible_devices=0,1,2,3,4,5,6,7

NNODES=${NNODES:=1}
NPROC_PER_NODE=${NPROC_PER_NODE:=8}
NODE_RANK=${NODE_RANK:=0}
MASTER_ADDR=${MASTER_ADDR:=0.0.0.0}
MASTER_PORT=${MASTER_PORT:=24001}


torchrun --nnodes=$NNODES --nproc-per-node $NPROC_PER_NODE --node-rank $NODE_RANK \
  --master-addr=$MASTER_ADDR --master-port=$MASTER_PORT $@ 2>&1 | tee log.txt

#!/bin/bash

ARGS=""
if [ ! -z "$NUM_GPU" ]; then
  ARGS="--gpu="
  for i in $(seq 0 $((NUM_GPU-1)))
  do
    ARGS="$ARGS$i,"
  done

  sed -i "s/batch_size: 4/batch_size: $NUM_GPU/g" ./configs/stable-diffusion/v1-finetune-4gpu.yaml
  sed -i "s/num_workers: 4/num_workers: $NUM_GPU/g" ./configs/stable-diffusion/v1-finetune-4gpu.yaml
fi

python3 main.py $ARGS "$@"

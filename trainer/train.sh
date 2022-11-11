#!/bin/bash

# Just an example of how to run the training script.

export HF_API_TOKEN="your_token"
BASE_MODEL="runwayml/stable-diffusion-v1-5"
RUN_NAME="artstation-4-A6000"
DATASET="/mnt/sd-finetune-data/artstation-dataset-full"
N_GPU=4
N_EPOCHS=2
BATCH_SIZE=4

python3 -m torch.distributed.run --nproc_per_node=$N_GPU diffusers_trainer.py --model=$BASE_MODEL --run_name=$RUN_NAME --dataset=$DATASET --bucket_side_min=64 --use_8bit_adam=True --gradient_checkpointing=True --batch_size=$BATCH_SIZE --fp16=True --image_log_steps=500 --epochs=$N_EPOCHS --resolution=768 --use_ema=True --clip_penultimate=False

# and to resume... just add the --resume flag and supply it with the path to the checkpoint.
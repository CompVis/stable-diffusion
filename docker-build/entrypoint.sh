#!/bin/bash
ln -sf /data/sd-v1-4.ckpt /stable-diffusion/models/ldm/stable-diffusion-v1/model.ckpt
cd /stable-diffusion
conda activate ldm

if [ $# -eq 0 ]; then
    python3 scripts/dream.py --full_precision -o /data 
else
    python3 scripts/dream.py --full_precision -o /data "$@"
fi

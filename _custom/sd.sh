#!/usr/bin/bash

#conda activate ldm

python scripts/txt2img.py --plms --ckpt sd-v1-4.ckpt --skip_grid --n_samples 3 --outdir /mnt/c/Users/chris/Desktop/ --seed 17 --prompt "${@}"

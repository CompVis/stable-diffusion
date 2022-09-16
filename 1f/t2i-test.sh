set -e
python3 scripts/txt2img.py --prompt "a drawing of a giraffe riding a bicycle in space" --plms --ckpt ./weights/sd-v1-4.ckpt --skip_grid --n_samples 1 --n_iter 1

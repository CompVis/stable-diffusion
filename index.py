#!/home/joe/miniconda3/envs/stablediffusion/bin/python
import os
import random

styles = [
    "an old photograph",
    "an old polaroid",
    "a polaroid",
    "a photograph",
    "a painting",
    "a watercolor",
    "a sketch",
    "a drawing",
    "a charcoal drawing",
    "a charcoal sketch",
    "a charcoal painting",
    "a charcoal watercolor",
    "a charcoal watercolor painting",
]

style = random.choice(styles)

prompt = f"{style} of a bird"
command = f'python scripts/txt2img.py --prompt "" --plms --n_samples 1'
os.system(command)
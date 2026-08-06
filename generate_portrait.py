#!/usr/bin/env python3
"""
Generate a hot woman portrait using Stable Diffusion with optimizations for CPU
"""

import torch
from diffusers import StableDiffusionPipeline
import os

os.makedirs("outputs", exist_ok=True)

print("Loading Stable Diffusion model (optimized for CPU)...")
device = "cpu"

# Use a smaller, optimized model and settings for faster CPU generation
model_id = "runwayml/stable-diffusion-v1-5"

pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float32,
    safety_checker=None,  # Disable safety checker to speed up
    requires_safety_checker=False
)

pipe = pipe.to(device)
pipe.enable_attention_slicing()  # Reduce memory usage for CPU

prompt = "a beautiful hot woman portrait, professional photography, studio lighting, detailed face, 8k, high quality"

print(f"\nGenerating image with prompt: '{prompt}'")
print("Running inference (this will take 10-15 minutes on CPU)...\n")

image = pipe(
    prompt,
    num_inference_steps=20,  # Reduced steps for faster generation
    guidance_scale=7.5,
    height=512,
    width=512
).images[0]

output_path = "outputs/hot_woman_portrait.png"
image.save(output_path)

print(f"✓ Image saved to: {output_path}")

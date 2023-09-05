import os
import time

from typing import List
from PIL import Image
from random import randint

from scripts.retro_diffusion import rd
from scripts.util.audio import play
from scripts.util.clbar import clbar
from scripts.util.lora import load_loras, prepare_loras_for_inference, restore_loras_after_inference
from scripts.util.palettize import palettizeOutput
from scripts.util.post_inference_processing import image_postprocessing

from sdkit.generate import generate_images as sdkit_generate_images

def img2img(
    loraPath: str,
    loraFiles: List[str],
    loraWeights: List[int],
    device: str,
    precision: str,
    pixelSize: int,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    inference_steps: int,
    guidance_scale: float,
    denoising_strength: float,
    seed: int,
    n_iter: int,
    tilingX: str,
    tilingY: str,
    pixelvae: bool,
    postprocess: bool,
):
    timer = time.time()
    
    assert 0.0 <= denoising_strength <= 1.0, "can only work with strength in [0.0, 1.0]"

    # Calculate the number of steps for encoding
    denoising_steps = int(denoising_strength * inference_steps)
    
    init_img = "temp/input.png"

    # Load initial image and move it to the specified device
    assert os.path.isfile(init_img)
    init_image = Image.open(init_img)   

    os.makedirs("temp", exist_ok=True)
    outpath = "temp"

    # Set a random seed if not provided
    if seed == None:
        seed = randint(0, 1000000)

    rd.logger(
        f"\n[#48a971]Image to Image[white] generating for [#48a971]{n_iter}[white] iterations with [#48a971]{denoising_steps}[white] steps per iteration at [#48a971]{width}[white]x[#48a971]{height}"
    )

    assert prompt is not None

    # Prepare loras for inference
    prepare_loras_for_inference(loraPath, loraFiles, loraWeights)
    
    # Load loras
    load_loras(rd.loras)

    if pixelvae:
        print("TODO: Implement pixelvae")

    seeds = []

    base_count = 1
    # Iterate over the specified number of iterations
    for _ in clbar(
        range(n_iter),
        name="Iterations",
        position="last",
        unit="image",
        prefixwidth=12,
        suffixwidth=28,
    ):
        images = sdkit_generate_images(
            rd.context,
            prompt=prompt,
            seed=seed,
            width=width,
            height=height,
            negative_prompt=negative_prompt,
            num_outputs=1,
            num_inference_steps=denoising_steps,
            sampler_name="euler",
            guidance_scale=guidance_scale,
            lora_alpha=rd.lora_alpha,
            init_image=init_image,
        )

        for image in images:
            file_name = "temp" + f"{base_count}"
            
            is_1bit = "1bit.pxlm" in loraFiles
            
            postprocessed_image = image_postprocessing(image, width, height, pixelSize, is_1bit)

            postprocessed_image.save(os.path.join(outpath, file_name + ".png"))

            if n_iter > 1 and base_count < n_iter:
                play("iteration.wav")

            seeds.append(str(seed))
            seed += 1
            base_count += 1
            
    # Release loras
    restore_loras_after_inference(rd.loras)

    if postprocess:
        play("iteration.wav")
        palettizeOutput(int(n_iter))
    else:
        play("batch.wav")
    rd.logger(
        f"[#c4f129]Image generation completed in [#48a971]{round(time.time()-timer, 2)} [#c4f129]seconds\n[#48a971]Seeds: [#494b9b]{', '.join(seeds)}"
    )
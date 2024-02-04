import torch
import time
import os
import math

from random import randint
from ldm.util import max_tile
from torch import autocast
from contextlib import nullcontext
from cryptography.fernet import Fernet
from lora import load_lora_raw

from ldm.controlnet import load_controlnet
from ldm.apply_controlnet import load_image, apply_controlnet
from ldm.sample import prepare_noise, sample

# TODO: Move this SD model to our own
from ldm.model_detection import model_config_from_unet
from ldm.model_patcher import ModelPatcher
from ldm.model_management import load_model_gpu
from ldm.lora import load_lora_for_models


def load_controlnet(
    controlnet_model,
    conditioning_img,
    strength,
    state_dict,
    device,
    conditioning,
    negative_conditioning,
    raw_loras=[],
    unet_dtype=torch.float16,
):
    # Load controlnet model
    controlnet = load_controlnet(controlnet_model)

    # Load conditioning image
    (image,) = load_image(conditioning_img)

    # Create controlnet model
    model_config = model_config_from_unet(
        state_dict, "model.diffusion_model.", unet_dtype
    )

    # Set the weights
    sd_model = model_config.get_model(
        state_dict,
        "model.diffusion_model.",
        device=device,
    )
    sd_model.load_model_weights(state_dict, "model.diffusion_model.")

    # Create the comfy model
    model_patcher = ModelPatcher(
        sd_model,
        load_device=device,
        current_device=device,
        offload_device=torch.device("cpu"),
    )

    # Move model to GPU
    load_model_gpu(model_patcher)

    # Apply loras
    lora_model_patcher = model_patcher

    for lora in raw_loras:
        lora_model_patcher, _clip = load_lora_for_models(
            lora_model_patcher, None, lora["sd"], lora["weight"] / 100, 0
        )

    # Compute conditioning
    cldm_conditioning = [[conditioning[0], {"pooled_output": None}]]
    cldm_negative_conditioning = [[negative_conditioning[0], {"pooled_output": None}]]

    # Apply controlnet to conditioning
    (controlled_conditioning,) = apply_controlnet(
        cldm_conditioning, controlnet, image, strength
    )

    return lora_model_patcher, controlled_conditioning, cldm_negative_conditioning


def sample_cldm(
    model_patcher,
    conditioning,
    negative_conditioning,
    seed,
    steps = 20,
    cfg = 5.0,
    sampler = "euler",
    batch=1,
    width=512,
    height=512,
    latent=None,
    scheduler = "normal",
):
    # Generate empty latents for txt2img
    if latent is None:
        latent = torch.zeros([batch, 4, height // 8, width // 8])

    # Prepare noise
    noise = prepare_noise(latent, seed, None)
    
    samples_cldm = sample(
        model_patcher,
        noise,
        steps,
        cfg,
        sampler,
        scheduler,
        conditioning,
        negative_conditioning,
        latent,
        seed=seed,
    )
    
    # Scale the latents to make them compatible with RD decoding
    samples_cldm /= 10.0
    
    return samples_cldm

def prepare_cldm(
    prompt,
    negative,
    translate,
    promptTuning,
    W,
    H,
    pixelSize,
    upscale,
    quality,
    scale,
    lighting,
    composition,
    seed,
    total_images,
    maxBatchSize,
    device,
    precision,
    loras,
    tilingX,
    tilingY,
    preview,
    pixelvae,
    post,
    
    # Internal functions
    rprint,
    manageComposition,
    managePrompts,
    seed_everything,
    modelPath,
):
    timer = time.time()

    # Check gpu availability
    if device == "cuda" and not torch.cuda.is_available():
        if torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
            rprint(f"\n[#ab333d]GPU is not responding, loading model in CPU mode")

    # Calculate maximum batch size
    global maxSize
    maxSize = maxBatchSize
    size = math.sqrt(W * H)
    if size >= maxSize or device == "cpu":
        batch = 1
    else:
        batch = min(total_images, math.floor((maxSize / size) ** 2))
    runs = (
        math.floor(total_images / batch)
        if total_images % batch == 0
        else math.floor(total_images / batch) + 1
    )

    # Set the seed for random number generation if not provided
    if seed == None:
        seed = randint(0, 1000000)

    # Set attention map tile values
    wtile = max_tile(W // 8)
    htile = max_tile(H // 8)

    # Derive steps, cfg, lcm weight from quality setting
    # Curves defined by https://www.desmos.com/calculator/aazom0lzyz
    steps = round(3.4 + ((quality**2) / 1.5))
    scale = max(1, scale * ((1.6 + (((quality - 1.6) ** 2) / 4)) / 5))
    lcm_weight = max(1.5, 10 - (quality * 1.5))
    if lcm_weight > 0:
        loras.append(
            {
                "file": os.path.join(modelPath, "quality.lcm"),
                "weight": round(lcm_weight * 10),
            }
        )

    # High resolution adjustments for consistency
    gWidth = W // 8
    gHeight = H // 8

    if gWidth >= 96 or gHeight >= 96:
        loras.append({"file": os.path.join(modelPath, "resfix.lcm"), "weight": 40})

    # Composition and lighting modifications
    loras = manageComposition(lighting, composition, loras)

    # Composition enhancement settings (high res fix)
    pre_steps = steps
    up_steps = 1
    if gWidth >= 96 and gHeight >= 96 and upscale:
        lower = 50
        aspect = gWidth / gHeight
        gx = gWidth
        gy = gHeight
        # Calculate initial image size from given large image
        # Targets resolutions between 64x64 and 96x96 while respecting aspect ratios
        # Interactive example here: https://editor.p5js.org/Astropulse/full/Co7CGTAnm
        gWidth = int((lower * max(1, aspect)) + ((gy / 7) * aspect))
        gHeight = int((lower * max(1, 1 / aspect)) + ((gx / 7) * (1 / aspect)))

        # Curves defined by https://www.desmos.com/calculator/aazom0lzyz
        pre_steps = round(steps * ((10 - (((quality - 1.1) ** 2) / 6)) / 10))
        up_steps = round(steps * (((((quality - 6.5) ** 2) / 1.6) + 2.4) / 10))
    else:
        upscale = False

    # Apply modifications to raw prompts
    data, negative_data = managePrompts(
        prompt,
        negative,
        W,
        H,
        seed,
        upscale,
        total_images,
        loras,
        translate,
        promptTuning,
    )
    seed_everything(seed)

    rprint(
        f"\n[#48a971]Text to Image[white] generating [#48a971]{total_images}[white] quality [#48a971]{quality}[white] images over [#48a971]{runs}[white] batches with [#48a971]{wtile}[white]x[#48a971]{htile}[white] attention tiles at [#48a971]{W}[white]x[#48a971]{H}[white] ([#48a971]{W // pixelSize}[white]x[#48a971]{H // pixelSize}[white] pixels)"
    )

    if W // 8 >= 96 and H // 8 >= 96 and upscale:
        rprint(
            f"[#48a971]Pre-generating[white] composition image at [#48a971]{gWidth * 8}[white]x[#48a971]{gHeight * 8} [white]([#48a971]{(gWidth * 8) // pixelSize}[white]x[#48a971]{(gHeight * 8) // pixelSize}[white] pixels)"
        )

    # Set the precision scope based on device and precision
    if device == "cuda" and precision == "autocast":
        precision_scope = autocast
    else:
        precision_scope = nullcontext

    # !!! REMEMBER: ALL MODEL FILES ARE BOUND UNDER THE LICENSE AGREEMENTS OUTLINED HERE: https://astropulse.co/#retrodiffusioneula https://astropulse.co/#retrodiffusionmodeleula !!!
    raw_loras = []
    decryptedFiles = []
    fernet = Fernet("I47jl1hqUPug4KbVYd60_zeXhn_IH_ECT3QRGiBxdxo=")
    for i, loraPair in enumerate(loras):
        decryptedFiles.append("none")
        _, loraName = os.path.split(loraPair["file"])
        if loraName != "none":
            if os.path.splitext(loraName)[1] == ".pxlm":
                with open(loraPair["file"], "rb") as enc_file:
                    encrypted = enc_file.read()
                    try:
                        decryptedFiles[i] = fernet.decrypt(encrypted)
                    except:
                        decryptedFiles[i] = encrypted

                    with open(loraPair["file"], "wb") as dec_file:
                        dec_file.write(decryptedFiles[i])
                        try:
                            raw_loras.append(
                                {
                                    "sd": load_lora_raw(loraPair["file"]),
                                    "weight": loraPair["weight"],
                                }
                            )
                        except:
                            # Decrypted file could not be read, revert to unchanged, and return an error
                            decryptedFiles[i] = "none"
                            dec_file.write(encrypted)
                            rprint(
                                f"[#ab333d]Modifier {os.path.splitext(loraName)[0]} could not be loaded, the file may be corrupted"
                            )
                            continue
                        
            # Prepare for inference
            if not any(name == os.path.splitext(loraName)[0] for name in system_models):
                rprint(
                    f"[#494b9b]Using [#48a971]{os.path.splitext(loraName)[0]} [#494b9b]LoRA with [#48a971]{loraPair['weight']}% [#494b9b]strength"
                )
    
    

import torch

from ldm.controlnet import load_controlnet as load_controlnet_cldm
from ldm.apply_controlnet import load_image, apply_controlnet
from ldm.sample import prepare_noise, sample

from ldm.model_management import unload_all_models
from ldm.lora import load_lora_for_models
from ldm.sd import load_checkpoint_guess_config


def load_controlnet(
    controlnet_model,
    conditioning_img,
    strength,
    # state_dict,
    device,
    conditioning,
    negative_conditioning,
    raw_loras=[],
    unet_dtype=torch.float16,
):
    # Load controlnet model
    controlnet = load_controlnet_cldm(controlnet_model)

    # Load conditioning image
    (image, _mask) = load_image(conditioning_img)

    # Load base model
    out = load_checkpoint_guess_config(
        "./models/base/model.safetensors",
        output_vae=False,
        output_clip=False,
        output_clipvision=False,
    )
    
    model_patcher = out[0]
    
    # model_config = model_config_from_unet(
    #     state_dict, "model.diffusion_model.", unet_dtype
    # )

    # # Set the weights
    # sd_model = model_config.get_model(
    #     state_dict,
    #     "model.diffusion_model.",
    #     device=device,
    # )
    # sd_model.load_model_weights(state_dict, "model.diffusion_model.")

    # # Create the comfy model
    # model_patcher = ModelPatcher(
    #     sd_model,
    #     load_device=device,
    #     current_device=device,
    #     offload_device=torch.device("cpu"),
    # )

    # # Move model to GPU
    # load_model_gpu(model_patcher)

    # Apply loras
    lora_model_patcher = model_patcher

    for lora in raw_loras:
        print(lora)
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
    samples_cldm /= 6.0
    
    return samples_cldm

def unload_cldm():
    # Unload the model
    unload_all_models()
    
    return
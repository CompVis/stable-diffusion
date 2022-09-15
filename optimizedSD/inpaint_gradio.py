import argparse
import os
import re
import time
from contextlib import nullcontext
from itertools import islice
from random import randint

import gradio as gr
import numpy as np
import torch
from PIL import Image
from einops import rearrange, repeat
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from torch import autocast
from torchvision.utils import make_grid
from tqdm import tqdm, trange
from transformers import logging

from ldm.util import instantiate_from_config
from optimUtils import split_weighted_subprompts, logger

logging.set_verbosity_error()
import mimetypes

mimetypes.init()
mimetypes.add_type("application/javascript", ".js")


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    return sd


def load_img(image, h0, w0):
    image = image.convert("RGB")
    w, h = image.size
    print(f"loaded input image of size ({w}, {h})")
    if h0 is not None and w0 is not None:
        h, w = h0, w0

    w, h = map(lambda x: x - x % 64, (w, h))  # resize to integer multiple of 32

    print(f"New image size ({w}, {h})")
    image = image.resize((w, h), resample=Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0


def load_mask(mask, h0, w0, newH, newW, invert=False):
    image = mask.convert("RGB")
    w, h = image.size
    print(f"loaded input mask of size ({w}, {h})")
    if h0 is not None and w0 is not None:
        h, w = h0, w0

    w, h = map(lambda x: x - x % 64, (w, h))  # resize to integer multiple of 32

    print(f"New mask size ({w}, {h})")
    image = image.resize((newW, newH), resample=Image.LANCZOS)
    # image = image.resize((64, 64), resample=Image.LANCZOS)
    image = np.array(image)

    if invert:
        print("inverted")
        where_0, where_1 = np.where(image == 0), np.where(image == 255)
        image[where_0], image[where_1] = 255, 0
    image = image.astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return image


def generate(
        image,
        mask_image,
        prompt,
        strength,
        ddim_steps,
        n_iter,
        batch_size,
        Height,
        Width,
        scale,
        ddim_eta,
        unet_bs,
        device,
        seed,
        outdir,
        img_format,
        turbo,
        full_precision,
):
    if seed == "":
        seed = randint(0, 1000000)
    seed = int(seed)
    seed_everything(seed)
    sampler = "ddim"

    # Logging
    logger(locals(), log_csv="logs/inpaint_gradio_logs.csv")

    init_image = load_img(image['image'], Height, Width).to(device)

    model.unet_bs = unet_bs
    model.turbo = turbo
    model.cdevice = device
    modelCS.cond_stage_model.device = device

    if device != "cpu" and full_precision == False:
        model.half()
        modelCS.half()
        modelFS.half()
        init_image = init_image.half()
        # mask.half()

    tic = time.time()
    os.makedirs(outdir, exist_ok=True)
    outpath = outdir
    sample_path = os.path.join(outpath, "_".join(re.split(":| ", prompt)))[:150]
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))

    # n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
    assert prompt is not None
    data = [batch_size * [prompt]]

    modelFS.to(device)

    init_latent = modelFS.get_first_stage_encoding(modelFS.encode_first_stage(init_image))  # move to latent space
    init_latent = repeat(init_latent, "1 ... -> b ...", b=batch_size)
    if mask_image is None:
        mask = load_mask(image['mask'], Height, Width, init_latent.shape[2], init_latent.shape[3], True).to(device)
    else:
        image['mask']=mask_image
        mask = load_mask(mask_image, Height, Width, init_latent.shape[2], init_latent.shape[3], True).to(device)
    
    mask = mask[0][0].unsqueeze(0).repeat(4, 1, 1).unsqueeze(0)
    mask = repeat(mask, '1 ... -> b ...', b=batch_size)

    if device != "cpu":
        mem = torch.cuda.memory_allocated() / 1e6
        modelFS.to("cpu")
        while torch.cuda.memory_allocated() / 1e6 >= mem:
            time.sleep(1)

    if strength == 1:
        print("strength should be less than 1, setting it to 0.999")
        strength = 0.999
    assert 0.0 <= strength < 1.0, "can only work with strength in [0.0, 1.0]"
    t_enc = int(strength * ddim_steps)
    print(f"target t_enc is {t_enc} steps")

    if full_precision == False and device != "cpu":
        precision_scope = autocast
    else:
        precision_scope = nullcontext

    all_samples = []
    seeds = ""
    with torch.no_grad():
        all_samples = list()
        for _ in trange(n_iter, desc="Sampling"):
            for prompts in tqdm(data, desc="data"):
                with precision_scope("cuda"):
                    modelCS.to(device)
                    uc = None
                    if scale != 1.0:
                        uc = modelCS.get_learned_conditioning(batch_size * [""])
                    if isinstance(prompts, tuple):
                        prompts = list(prompts)

                    subprompts, weights = split_weighted_subprompts(prompts[0])
                    if len(subprompts) > 1:
                        c = torch.zeros_like(uc)
                        totalWeight = sum(weights)
                        # normalize each "sub prompt" and add it
                        for i in range(len(subprompts)):
                            weight = weights[i]
                            # if not skip_normalize:
                            weight = weight / totalWeight
                            c = torch.add(c, modelCS.get_learned_conditioning(subprompts[i]), alpha=weight)
                    else:
                        c = modelCS.get_learned_conditioning(prompts)

                    if device != "cpu":
                        mem = torch.cuda.memory_allocated() / 1e6
                        modelCS.to("cpu")
                        while torch.cuda.memory_allocated() / 1e6 >= mem:
                            time.sleep(1)

                    # encode (scaled latent)
                    z_enc = model.stochastic_encode(
                        init_latent, torch.tensor([t_enc] * batch_size).to(device),
                        seed, ddim_eta, ddim_steps)

                    # decode it
                    samples_ddim = model.sample(
                        t_enc,
                        c,
                        z_enc,
                        unconditional_guidance_scale=scale,
                        unconditional_conditioning=uc,
                        mask=mask,
                        x_T=init_latent,
                        sampler=sampler,
                    )

                    modelFS.to(device)
                    print("saving images")
                    for i in range(batch_size):
                        x_samples_ddim = modelFS.decode_first_stage(samples_ddim[i].unsqueeze(0))
                        x_sample = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                        all_samples.append(x_sample.to("cpu"))
                        x_sample = 255.0 * rearrange(x_sample[0].cpu().numpy(), "c h w -> h w c")
                        Image.fromarray(x_sample.astype(np.uint8)).save(
                            os.path.join(sample_path, "seed_" + str(seed) + "_" + f"{base_count:05}.{img_format}")
                        )
                        seeds += str(seed) + ","
                        seed += 1
                        base_count += 1

                    if device != "cpu":
                        mem = torch.cuda.memory_allocated() / 1e6
                        modelFS.to("cpu")
                        while torch.cuda.memory_allocated() / 1e6 >= mem:
                            time.sleep(1)

                    del samples_ddim
                    del x_sample
                    del x_samples_ddim
                    print("memory_final = ", torch.cuda.memory_allocated() / 1e6)

    toc = time.time()

    time_taken = (toc - tic) / 60.0
    grid = torch.cat(all_samples, 0)
    grid = make_grid(grid, nrow=n_iter)
    grid = 255.0 * rearrange(grid, "c h w -> h w c").cpu().numpy()

    txt = (
            "Samples finished in "
            + str(round(time_taken, 3))
            + " minutes and exported to \n"
            + sample_path
            + "\nSeeds used = "
            + seeds[:-1]
    )
    return Image.fromarray(grid.astype(np.uint8)), image['mask'], txt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='txt2img using gradio')
    parser.add_argument('--config_path', default="optimizedSD/v1-inference.yaml", type=str, help='config path')
    parser.add_argument('--ckpt_path', default="models/ldm/stable-diffusion-v1/model.ckpt", type=str, help='ckpt path')
    args = parser.parse_args()
    config = args.config_path
    ckpt = args.ckpt_path
    sd = load_model_from_config(f"{ckpt}")
    li, lo = [], []
    for key, v_ in sd.items():
        sp = key.split(".")
        if (sp[0]) == "model":
            if "input_blocks" in sp:
                li.append(key)
            elif "middle_block" in sp:
                li.append(key)
            elif "time_embed" in sp:
                li.append(key)
            else:
                lo.append(key)
    for key in li:
        sd["model1." + key[6:]] = sd.pop(key)
    for key in lo:
        sd["model2." + key[6:]] = sd.pop(key)

    config = OmegaConf.load(f"{config}")

    model = instantiate_from_config(config.modelUNet)
    _, _ = model.load_state_dict(sd, strict=False)
    model.eval()

    modelCS = instantiate_from_config(config.modelCondStage)
    _, _ = modelCS.load_state_dict(sd, strict=False)
    modelCS.eval()

    modelFS = instantiate_from_config(config.modelFirstStage)
    _, _ = modelFS.load_state_dict(sd, strict=False)
    modelFS.eval()
    del sd

    demo = gr.Interface(
        fn=generate,
        inputs=[
            gr.Image(tool="sketch", type="pil"),
            gr.Image(tool="editor", type="pil"),
            "text",
            gr.Slider(0, 0.99, value=0.99, step=0.01),
            gr.Slider(1, 1000, value=50),
            gr.Slider(1, 100, step=1),
            gr.Slider(1, 100, step=1),
            gr.Slider(64, 4096, value=512, step=64),
            gr.Slider(64, 4096, value=512, step=64),
            gr.Slider(0, 50, value=7.5, step=0.1),
            gr.Slider(0, 1, step=0.01),
            gr.Slider(1, 2, value=1, step=1),
            gr.Text(value="cuda"),
            "text",
            gr.Text(value="outputs/inpaint-samples"),
            gr.Radio(["png", "jpg"], value='png'),
            "checkbox",
            "checkbox",
        ],
        outputs=["image", "image", "text"],
    )
    demo.launch()

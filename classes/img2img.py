import os
import argparse
import random
import time

import numpy as np
import torch
from torch import autocast
from pytorch_lightning import seed_everything
from PIL import Image
from contextlib import  nullcontext
from tqdm import tqdm, trange
from classes.base import BaseModel

# import img2img functions from stable diffusion
from scripts.txt2img import make_grid
from einops import rearrange, repeat
from scripts.img2img import load_img


class Img2Img(BaseModel):
    args = [
        {
            "arg": "prompt",
            "type": str,
            "nargs": "?",
            "default": "a painting of a virus monster playing guitar",
            "help": "the prompt to render"
        },
        {
            "arg": "init-img",
            "type": str,
            "nargs": "?",
            "help": "path to the input image"
        },
        {
            "arg": "outdir",
            "type": str,
            "nargs": "?",
            "help": "dir to write results to",
            "default": "outputs/img2img-samples"
        },
        {
            "arg": "skip_grid",
            "action": "store_true",
            "help": "do not save a grid, only individual samples. Helpful when evaluating lots of samples",
        },
        {
            "arg": "skip_save",
            "action": "store_true",
            "help": "do not save indiviual samples. For speed measurements.",
        },
        {
            "arg": "ddim_steps",
            "type": int,
            "default": 50,
            "help": "number of ddim sampling steps",
        },
        {
            "arg": "plms",
            "action": "store_true",
            "help": "use plms sampling",
        },
        {
            "arg": "fixed_code",
            "action": "store_true",
            "help": "if enabled, uses the same starting code across all samples ",
        },
        {
            "arg": "ddim_eta",
            "type": float,
            "default": 0.0,
            "help": "ddim eta (eta=0.0 corresponds to deterministic sampling",
        },
        {
            "arg": "n_iter",
            "type": int,
            "default": 1,
            "help": "sample this often",
        },
        {
            "arg": "C",
            "type": int,
            "default": 4,
            "help": "latent channels",
        },
        {
            "arg": "f",
            "type": int,
            "default": 8,
            "help": "downsampling factor, most often 8 or 16",
        },
        {
            "arg": "n_samples",
            "type": int,
            "default": 2,
            "help": "how many samples to produce for each given prompt. A.k.a batch size",
        },
        {
            "arg": "n_rows",
            "type": int,
            "default": 0,
            "help": "rows in the grid (default: n_samples)",
        },
        {
            "arg": "scale",
            "type": float,
            "default": 5.0,
            "help": "unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
        },
        {
            "arg": "strength",
            "type": float,
            "default": 0.75,
            "help": "strength for noising/unnoising. 1.0 corresponds to full destruction of information in init image",
        },
        {
            "arg": "from-file",
            "type": str,
            "help": "if specified, load prompts from this file",
        },
        {
            "arg": "config",
            "type": str,
            "default": "configs/stable-diffusion/v1-inference.yaml",
            "help": "path to config which constructs model",
        },
        {
            "arg": "ckpt",
            "type": str,
            "default": "models/ldm/stable-diffusion-v1/model.ckpt",
            "help": "path to checkpoint of model",
        },
        {
            "arg": "seed",
            "type": int,
            "default": random.randint(0, 100000),
            # default=42,
            "help": "the seed (for reproducible sampling)",
        },
        {
            "arg": "precision",
            "type": str,
            "help": "evaluate at this precision",
            "choices": ["full", "autocast"],
            "default": "autocast"
        }
    ]

    def parse_arguments(self):
        self.parse_arguments()
        parser = argparse.ArgumentParser()
        opt = parser.parse_args()
        self.opt = opt

    def sample(self, options=None):
        super().sample(options)
        opt = self.opt
        batch_size = self.batch_size
        model = self.model
        sampler = self.sampler
        data = self.data
        sample_path = self.sample_path
        base_count = self.base_count
        n_rows = self.n_rows
        device = self.device
        outpath = self.outpath
        grid_count = self.grid_count
        seed = opt.seed

        assert os.path.isfile(opt.init_img)
        init_image = load_img(opt.init_img).to(device)
        init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
        init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))  # move to latent space

        sampler.make_schedule(ddim_num_steps=opt.ddim_steps, ddim_eta=opt.ddim_eta, verbose=False)

        assert 0. <= opt.strength <= 1., 'can only work with strength in [0.0, 1.0]'
        t_enc = int(opt.strength * opt.ddim_steps)
        print(f"target t_enc is {t_enc} steps")

        precision_scope = autocast if opt.precision == "autocast" else nullcontext
        with torch.no_grad():
            with precision_scope("cuda"):
                with model.ema_scope():
                    tic = time.time()
                    all_samples = list()
                    for n in trange(opt.n_iter, desc="Sampling"):
                        for prompts in tqdm(data, desc="data"):
                            uc = None
                            if opt.scale != 1.0:
                                uc = model.get_learned_conditioning(batch_size * [""])
                            if isinstance(prompts, tuple):
                                prompts = list(prompts)
                            c = model.get_learned_conditioning(prompts)

                            # encode (scaled latent)
                            z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc] * batch_size).to(device))
                            # decode it
                            samples = sampler.decode(z_enc, c, t_enc, unconditional_guidance_scale=opt.scale,
                                                     unconditional_conditioning=uc, )

                            x_samples = model.decode_first_stage(samples)
                            x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                            if not opt.skip_save:
                                for x_sample in x_samples:
                                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                    Image.fromarray(x_sample.astype(np.uint8)).save(
                                        os.path.join(sample_path, f"{base_count:05}.png"))
                                    base_count += 1

                            all_samples.append(x_samples)

                    if not opt.skip_grid:
                        # additionally, save as grid
                        grid = torch.stack(all_samples, 0)
                        grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                        grid = make_grid(grid, nrow=n_rows)

                        # to image
                        grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                        Image.fromarray(grid.astype(np.uint8)).save(os.path.join(outpath, f'grid-{grid_count:04}.png'))
                        grid_count += 1

                    toc = time.time()

        print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
              f" \nEnjoy.")
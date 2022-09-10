import os
from classes.base import BaseModel
import numpy as np
import time
import torch
from torch import autocast
from pytorch_lightning import seed_everything
from PIL import Image
from tqdm import tqdm, trange
from einops import rearrange
from contextlib import  nullcontext

# import txt2img functions from stable diffusion
from scripts.txt2img import check_safety
from scripts.txt2img import make_grid


class Txt2Img(BaseModel):
    args = [
        {
            "arg": "prompt",
            "type": str,
            "nargs": "?",
            "default": "a painting of a virus monster playing guitar",
            "help": "the prompt to render"
        },
        {
            "arg": "outdir",
            "type": str,
            "nargs": "?",
            "help": "dir to write results to",
            "default": "outputs/txt2img-samples"
        },
        {
            "arg": "skip_grid",
            "action": "store_true",
            "help": "do not save a grid, only individual samples. Helpful when evaluating lots of samples",
        },
        {
            "arg": "skip_save",
            "action": "store_true",
            "help": "do not save individual samples. For speed measurements.",
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
            "arg": "laion400m",
            "action": "store_true",
            "help": "uses the LAION400M model",
        },
        {
            "arg": "fixed_code",
            "action": "store_true",
            "help": "if enabled, uses the same starting code across samples ",
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
            "default": 2,
            "help": "sample this often",
        },
        {
            "arg": "H",
            "type": int,
            "default": 512,
            "help": "image height, in pixel space",
        },
        {
            "arg": "W",
            "type": int,
            "default": 512,
            "help": "image width, in pixel space",
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
            "help": "downsampling factor",
        },
        {
            "arg": "n_samples",
            "type": int,
            "default": 3,
            "help": "how many samples to produce for each given prompt. A.k.a. batch size",
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
            "default": 7.5,
            "help": "unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
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
            "default": 42,
            "help": "the seed (for reproducible sampling)",
        },
        {
            "arg": "precision",
            "type": str,
            "help": "evaluate at this precision",
            "choices": ["full", "autocast"],
            "default": "autocast"
        },
        {
            "arg": "out_folder",
            "type": str,
            "help": "where to store pics",
            "default": ""
        },
    ]

    def sample(self, options):
        super().sample(options)
        opt = self.opt
        model = self.model
        sample_path = self.sample_path
        n_rows = self.n_rows
        outpath = self.outpath
        grid_count = self.grid_count
        data = self.data
        batch_size = self.batch_size
        sampler = self.sampler
        start_code = self.start_code
        base_count = self.base_count
        seed = opt.seed

        precision_scope = autocast if opt.precision=="autocast" else nullcontext

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
                            shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                            samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                             conditioning=c,
                                                             batch_size=opt.n_samples,
                                                             shape=shape,
                                                             verbose=False,
                                                             unconditional_guidance_scale=opt.scale,
                                                             unconditional_conditioning=uc,
                                                             eta=opt.ddim_eta,
                                                             x_T=start_code)

                            x_samples_ddim = model.decode_first_stage(samples_ddim)
                            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                            x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

                            x_checked_image, has_nsfw_concept = check_safety(x_samples_ddim)

                            x_checked_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)

                            if not opt.skip_save:
                                for x_sample in x_checked_image_torch:
                                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                    img = Image.fromarray(x_sample.astype(np.uint8))
                                    img = self.put_watermark(img, self.wm_encoder)
                                    img.save(os.path.join(sample_path, f"{base_count:05}.png"))
                                    base_count += 1

                            if not opt.skip_grid:
                                all_samples.append(x_checked_image_torch)

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



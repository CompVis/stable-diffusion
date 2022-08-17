"""Simplified text to image API for stable diffusion/latent diffusion

Example Usage:

from ldm.simplet2i import T2I
# Create an object with default values
t2i = T2I(outdir      = <path>        // outputs/txt2img-samples
          model       = <path>        // models/ldm/stable-diffusion-v1/model.ckpt
          config      = <path>        // default="configs/stable-diffusion/v1-inference.yaml
          batch       = <integer>     // 1
          steps       = <integer>     // 50
          seed        = <integer>     // current system time
          sampler     = ['ddim','plms']  // ddim
          grid        = <boolean>     // false
          width       = <integer>     // image width, multiple of 64 (512)
          height      = <integer>     // image height, multiple of 64 (512)
          cfg_scale   = <float>       // unconditional guidance scale (7.5)
          fixed_code  = <boolean>     // False
          )
# do the slow model initialization
t2i.load_model()

# Do the fast inference & image generation. Any options passed here 
# override the default values assigned during class initialization
# Will call load_model() if the model was not previously loaded.
t2i.txt2img(prompt = <string>           // required
            // the remaining option arguments override constructur value when present
            outdir = <path>             
            iterations  = <integer>
            batch       = <integer>
            steps       = <integer>
            seed        = <integer>     
            sampler     = ['ddim','plms']
            grid        = <boolean>
            width       = <integer> 
            height      = <integer>
            cfg_scale   = <float>
            ) -> boolean
                 
"""
import torch
import numpy as np
import random
import sys
import os
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext
import time
import math

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

class T2I:
    """T2I class
    Attributes
    ----------
    outdir
    model
    config
    iterations
    batch
    steps
    seed
    sampler
    grid
    individual
    width
    height
    cfg_scale
    fixed_code
    latent_channels
    downsampling_factor
    precision
"""
    def __init__(self,
                 outdir="outputs/txt2img-samples",
                 batch=1,
                 iterations = 1,
                 width=512,
                 height=512,
                 grid=False,
                 individual=None, # redundant
                 steps=50,
                 seed=None,
                 cfg_scale=7.5,
                 weights="models/ldm/stable-diffusion-v1/model.ckpt",
                 config = "configs/latent-diffusion/txt2img-1p4B-eval.yaml",
                 sampler="plms",
                 latent_channels=4,
                 downsampling_factor=8,
                 ddim_eta=0.0,  # deterministic
                 fixed_code=False,
                 precision='autocast'
    ):
        self.outdir     = outdir
        self.batch      = batch
        self.iterations = iterations
        self.width      = width
        self.height     = height
        self.grid       = grid
        self.steps      = steps
        self.cfg_scale  = cfg_scale
        self.weights   = weights
        self.config     = config
        self.sampler_name  = sampler
        self.fixed_code    = fixed_code
        self.latent_channels     = latent_channels
        self.downsampling_factor = downsampling_factor
        self.ddim_eta            = ddim_eta
        self.precision           = precision
        self.model      = None     # empty for now
        self.sampler    = None
        if seed is None:
            self.seed = self._new_seed()
        else:
            self.seed = seed
    def txt2img(self,prompt,outdir=None,batch=None,iterations=None,
                steps=None,seed=None,grid=None,individual=None,width=None,height=None,
                cfg_scale=None,ddim_eta=None):
        """ generate an image from the prompt, writing iteration images into the outdir """
        outdir     = outdir     or self.outdir
        steps      = steps      or self.steps
        seed       = seed       or self.seed
        width      = width      or self.width
        height     = height     or self.height
        cfg_scale  = cfg_scale  or self.cfg_scale
        ddim_eta   = ddim_eta   or self.ddim_eta
        batch      = batch or self.batch
        iterations = iterations or self.iterations

        model = self.load_model()  # will instantiate the model or return it from cache

        # grid and individual are mutually exclusive, with individual taking priority.
        # not necessary, but needed for compatability with dream bot
        if (grid is None):
            grid = self.grid
        if individual:
            grid = False
        
        data = [batch * [prompt]]

        # make directories and establish names for the output files
        os.makedirs(outdir, exist_ok=True)
        base_count = len(os.listdir(outdir))-1

        start_code = None
        if self.fixed_code:
            start_code = torch.randn([batch,
                                      self.latent_channels,
                                      height // self.downsampling_factor,
                                      width  // self.downsampling_factor],
                                     device=self.device)

        precision_scope = autocast if self.precision=="autocast" else nullcontext
        sampler         = self.sampler
        images = list()
        seeds  = list()

        tic    = time.time()
        
        with torch.no_grad():
            with precision_scope("cuda"):
                with model.ema_scope():
                    all_samples = list()
                    for n in trange(iterations, desc="Sampling"):
                        seed_everything(seed)
                        for prompts in tqdm(data, desc="data", dynamic_ncols=True):
                            uc = None
                            if cfg_scale != 1.0:
                                uc = model.get_learned_conditioning(batch * [""])
                            if isinstance(prompts, tuple):
                                prompts = list(prompts)
                            c = model.get_learned_conditioning(prompts)
                            shape = [self.latent_channels, height // self.downsampling_factor, width // self.downsampling_factor]
                            samples_ddim, _ = sampler.sample(S=steps,
                                                             conditioning=c,
                                                             batch_size=batch,
                                                             shape=shape,
                                                             verbose=False,
                                                             unconditional_guidance_scale=cfg_scale,
                                                             unconditional_conditioning=uc,
                                                             eta=ddim_eta,
                                                             x_T=start_code)

                            x_samples_ddim = model.decode_first_stage(samples_ddim)
                            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

                            if not grid:
                                for x_sample in x_samples_ddim:
                                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                    filename = os.path.join(outdir, f"{base_count:05}.png")
                                    Image.fromarray(x_sample.astype(np.uint8)).save(filename)
                                    images.append([filename,seed])
                                    base_count += 1
                            else:
                                all_samples.append(x_samples_ddim)
                                seeds.append(seed)

                        seed = self._new_seed()
 
                    if grid:
                        n_rows = batch if batch>1 else int(math.sqrt(batch * iterations))
                        # save as grid
                        grid = torch.stack(all_samples, 0)
                        grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                        grid = make_grid(grid, nrow=n_rows)

                        # to image
                        grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                        filename = os.path.join(outdir, f"{base_count:05}.png")
                        Image.fromarray(grid.astype(np.uint8)).save(filename)
                        for s in seeds:
                            images.append([filename,s])

        toc = time.time()
        print(f'{batch * iterations} images generated in',"%4.2fs"% (toc-tic))

        return images
        

    def _new_seed(self):
        self.seed = random.randrange(0,np.iinfo(np.uint32).max)
        return self.seed

    def load_model(self):
        """ Load and initialize the model from configuration variables passed at object creation time """
        if self.model is None:
            seed_everything(self.seed)
            try:
                config = OmegaConf.load(self.config)
                self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
                model = self._load_model_from_config(config,self.weights)
                self.model = model.to(self.device)
            except AttributeError:
                raise SystemExit

            if self.sampler_name=='plms':
                print("setting sampler to plms")
                self.sampler = PLMSSampler(self.model)
            elif self.sampler_name == 'ddim':
                print("setting sampler to ddim")
                self.sampler = DDIMSampler(self.model)
            else:
                print(f"unsupported sampler {self.sampler_name}, defaulting to plms")
                self.sampler = PLMSSampler(self.model)

        return self.model
                
    def _load_model_from_config(self, config, ckpt):
        print(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cpu")
        if "global_step" in pl_sd:
            print(f"Global Step: {pl_sd['global_step']}")
        sd = pl_sd["state_dict"]
        model = instantiate_from_config(config.model)
        m, u = model.load_state_dict(sd, strict=False)
        model.cuda()
        model.eval()
        return model


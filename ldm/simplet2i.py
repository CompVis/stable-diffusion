# Copyright (c) 2022 Lincoln D. Stein (https://github.com/lstein)

# Derived from source code carrying the following copyrights
# Copyright (c) 2022 Machine Vision and Learning Group, LMU Munich
# Copyright (c) 2022 Robin Rombach and Patrick Esser and contributors


"""Simplified text to image API for stable diffusion/latent diffusion

Example Usage:

from ldm.simplet2i import T2I
# Create an object with default values
t2i = T2I(outdir      = <path>        // outputs/txt2img-samples
          model       = <path>        // models/ldm/stable-diffusion-v1/model.ckpt
          config      = <path>        // default="configs/stable-diffusion/v1-inference.yaml
          iterations  = <integer>     // how many times to run the sampling (1)
          batch_size       = <integer>     // how many images to generate per sampling (1)
          steps       = <integer>     // 50
          seed        = <integer>     // current system time
          sampler_name= ['ddim', 'k_dpm_2_a', 'k_dpm_2', 'k_euler_a', 'k_euler', 'k_heun', 'k_lms', 'plms']  // k_lms
          grid        = <boolean>     // false
          width       = <integer>     // image width, multiple of 64 (512)
          height      = <integer>     // image height, multiple of 64 (512)
          cfg_scale   = <float>       // unconditional guidance scale (7.5)
          )

# do the slow model initialization
t2i.load_model()

# Do the fast inference & image generation. Any options passed here 
# override the default values assigned during class initialization
# Will call load_model() if the model was not previously loaded.
# The method returns a list of images. Each row of the list is a sub-list of [filename,seed]
results = t2i.txt2img(prompt = "an astronaut riding a horse"
                      outdir = "./outputs/txt2img-samples)
            )

for row in results:
    print(f'filename={row[0]}')
    print(f'seed    ={row[1]}')

# Same thing, but using an initial image.
results = t2i.img2img(prompt   = "an astronaut riding a horse"
                      outdir   = "./outputs/img2img-samples"
                      init_img = "./sketches/horse+rider.png")
                 
for row in results:
    print(f'filename={row[0]}')
    print(f'seed    ={row[1]}')
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
from einops import rearrange, repeat
from torchvision.utils import make_grid
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext
import time
import math
import re
import traceback

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim     import DDIMSampler
from ldm.models.diffusion.plms     import PLMSSampler
from ldm.models.diffusion.ksampler import KSampler

class T2I:
    """T2I class
    Attributes
    ----------
    model
    config
    iterations
    batch_size
    steps
    seed
    sampler_name
    width
    height
    cfg_scale
    latent_channels
    downsampling_factor
    precision
    strength
    embedding_path

The vast majority of these arguments default to reasonable values.
"""
    def __init__(self,
                 batch_size=1,
                 iterations = 1,
                 grid=False,
                 individual=None, # redundant
                 steps=50,
                 seed=None,
                 cfg_scale=7.5,
                 weights="models/ldm/stable-diffusion-v1/model.ckpt",
                 config = "configs/stable-diffusion/v1-inference.yaml",
                 sampler_name="klms",
                 latent_channels=4,
                 downsampling_factor=8,
                 ddim_eta=0.0,  # deterministic
                 precision='autocast',
                 full_precision=False,
                 strength=0.75, # default in scripts/img2img.py
                 embedding_path=None,
                 latent_diffusion_weights=False,  # just to keep track of this parameter when regenerating prompt
                 device='cuda'
    ):
        self.batch_size      = batch_size
        self.iterations = iterations
        self.width      = width
        self.height     = height
        self.grid       = grid
        self.steps      = steps
        self.cfg_scale  = cfg_scale
        self.weights    = weights
        self.config     = config
        self.sampler_name  = sampler_name
        self.latent_channels     = latent_channels
        self.downsampling_factor = downsampling_factor
        self.ddim_eta            = ddim_eta
        self.precision           = precision
        self.full_precision      = full_precision
        self.strength            = strength
        self.embedding_path      = embedding_path
        self.model      = None     # empty for now
        self.sampler    = None
        self.latent_diffusion_weights=latent_diffusion_weights
        self.device = device
        if seed is None:
            self.seed = self._new_seed()
        else:
            self.seed = seed

    def generate(self,
                 # these are common
                 prompt,
                 batch_size=None,
                 iterations=None,
                 steps=None,
                 seed=None,
                 cfg_scale=None,
                 ddim_eta=None,
                 skip_normalize=False,
                 image_callback=None,
                 # these are specific to txt2img
                 width=None,
                 height=None,
                 # these are specific to img2img
                 init_img=None,
                 strength=None,
                 variants=None):
        '''ldm.generate() is the common entry point for txt2img() and img2img()'''
        steps      = steps      or self.steps
        seed       = seed       or self.seed
        width      = width      or self.width
        height     = height     or self.height
        cfg_scale  = cfg_scale  or self.cfg_scale
        ddim_eta   = ddim_eta   or self.ddim_eta
        batch_size = batch_size or self.batch_size
        iterations = iterations or self.iterations
        strength   = strength   or self.strength

        model = self.load_model()  # will instantiate the model or return it from cache
        assert cfg_scale>1.0, "CFG_Scale (-C) must be >1.0"
        assert 0. <= strength <= 1., 'can only work with strength in [0.0, 1.0]'

        data = [batch_size * [prompt]]
        scope = autocast if self.precision=="autocast" else nullcontext
        if grid:
            callback = self.image2png
        else:
            callback = None

        tic    = time.time()
        if init_img:
            assert os.path.exists(init_img),f'{init_img}: File not found'
            results = self._img2img(prompt,
                                    data=data,precision_scope=scope,
                                    batch_size=batch_size,iterations=iterations,
                                    steps=steps,seed=seed,cfg_scale=cfg_scale,ddim_eta=ddim_eta,
                                    skip_normalize=skip_normalize,
                                    init_img=init_img,strength=strength,variants=variants,
                                    callback=image_callback)
        else:
            results = self._txt2img(prompt,
                                    data=data,precision_scope=scope,
                                    batch_size=batch_size,iterations=iterations,
                                    steps=steps,seed=seed,cfg_scale=cfg_scale,ddim_eta=ddim_eta,
                                    skip_normalize=skip_normalize,
                                    width=width,height=height,
                                    callback=image_callback)
        toc  = time.time()
        print(f'{len(results)} images generated in',"%4.2fs"% (toc-tic))
        return results
            
    @torch.no_grad()
    def _txt2img(self,prompt,
                 data,precision_scope,
                 batch_size,iterations,
                 steps,seed,cfg_scale,ddim_eta,
                 skip_normalize,
                 width,height,
                 callback=callback):    # the callback is called each time a new Image is generated
        """
        Generate an image from the prompt, writing iteration images into the outdir
        The output is a list of lists in the format: [[image1,seed1], [image2,seed2],...]
        """

        sampler         = self.sampler
        images = list()
        image_count = 0

        # Gawd. Too many levels of indent here. Need to refactor into smaller routines!
        try:
            with precision_scope(self.device.type), model.ema_scope():
                all_samples = list()
                for n in trange(iterations, desc="Sampling"):
                    seed_everything(seed)
                    for prompts in tqdm(data, desc="data", dynamic_ncols=True):
                        uc = None
                        if cfg_scale != 1.0:
                            uc = model.get_learned_conditioning(batch_size * [""])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)

                        # weighted sub-prompts
                        subprompts,weights = T2I._split_weighted_subprompts(prompts[0])
                        if len(subprompts) > 1:
                            # i dont know if this is correct.. but it works
                            c = torch.zeros_like(uc)
                            # get total weight for normalizing
                            totalWeight = sum(weights)
                            # normalize each "sub prompt" and add it
                            for i in range(0,len(subprompts)):
                                weight = weights[i]
                                if not skip_normalize:
                                    weight = weight / totalWeight
                                c = torch.add(c,model.get_learned_conditioning(subprompts[i]), alpha=weight)
                        else: # just standard 1 prompt
                            c = model.get_learned_conditioning(prompts)

                        shape = [self.latent_channels, height // self.downsampling_factor, width // self.downsampling_factor]
                        samples_ddim, _ = sampler.sample(S=steps,
                                                         conditioning=c,
                                                         batch_size=batch_size,
                                                         shape=shape,
                                                         verbose=False,
                                                         unconditional_guidance_scale=cfg_scale,
                                                         unconditional_conditioning=uc,
                                                         eta=ddim_eta)

                        x_samples_ddim = model.decode_first_stage(samples_ddim)
                        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                        for x_sample in x_samples_ddim:
                            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                            image = Image.fromarray(x_sample.astype(np.uint8))
                            images.append([image,seed])
                            if callback is not None:
                                callback(image,seed)
                                
                    seed = self._new_seed()
        except KeyboardInterrupt:
            print('*interrupted*')
            print('Partial results will be returned; if --grid was requested, nothing will be returned.')
        except RuntimeError as e:
            print(str(e))

        toc = time.time()
        print(f'{image_count} images generated in',"%4.2fs"% (toc-tic))
        return images
        
    @torch.no_grad()
    def _img2img(self,prompt,
                 data,precision_scope,
                 batch_size,iterations,
                 steps,seed,cfg_scale,ddim_eta,
                 skip_normalize,
                 init_img,strength,variants,
                 callback):
        """
        Generate an image from the prompt and the initial image, writing iteration images into the outdir
        The output is a list of lists in the format: [[image,seed1], [image,seed2],...]
        """

        # PLMS sampler not supported yet, so ignore previous sampler
        if self.sampler_name!='ddim':
            print(f"sampler '{self.sampler_name}' is not yet supported. Using DDM sampler")
            sampler = DDIMSampler(model, device=self.device)
        else:
            sampler = self.sampler

        init_image = self._load_img(init_img).to(self.device)
        init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
        with precision_scope(self.device.type):
            init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))  # move to latent space

        sampler.make_schedule(ddim_num_steps=steps, ddim_eta=ddim_eta, verbose=False)
        
        t_enc = int(strength * steps)
        print(f"target t_enc is {t_enc} steps")

        images = list()

        try:
            with precision_scope(self.device.type), model.ema_scope():
                all_samples = list()
                for n in trange(iterations, desc="Sampling"):
                    seed_everything(seed)
                    for prompts in tqdm(data, desc="data", dynamic_ncols=True):
                        uc = None
                        if cfg_scale != 1.0:
                            uc = model.get_learned_conditioning(batch_size * [""])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)

                        # weighted sub-prompts
                        subprompts,weights = T2I._split_weighted_subprompts(prompts[0])
                        if len(subprompts) > 1:
                            # i dont know if this is correct.. but it works
                            c = torch.zeros_like(uc)
                            # get total weight for normalizing
                            totalWeight = sum(weights)
                            # normalize each "sub prompt" and add it
                            for i in range(0,len(subprompts)):
                                weight = weights[i]
                                if not skip_normalize:
                                    weight = weight / totalWeight
                                c = torch.add(c,model.get_learned_conditioning(subprompts[i]), alpha=weight)
                        else: # just standard 1 prompt
                            c = model.get_learned_conditioning(prompts)

                        # encode (scaled latent)
                        z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc]*batch_size).to(self.device))
                        # decode it
                        samples = sampler.decode(z_enc, c, t_enc, unconditional_guidance_scale=cfg_scale,
                                                    unconditional_conditioning=uc,)

                        x_samples = model.decode_first_stage(samples)
                        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                        for x_sample in x_samples:
                            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                            image = Image.fromarray(x_sample.astype(np.uint8))
                            images.append([image,seed])
                            if callback is not None:
                                callback(image,seed)
                    seed = self._new_seed()

        except KeyboardInterrupt:
            print('*interrupted*')
            print('Partial results will be returned; if --grid was requested, nothing will be returned.')
        except RuntimeError as e:
            print("Oops! A runtime error has occurred. If this is unexpected, please copy-and-paste this stack trace and post it as an Issue to http://github.com/lstein/stable-diffusion")
            traceback.print_exc()
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
                self.device = torch.device(self.device) if torch.cuda.is_available() else torch.device("cpu")
                model = self._load_model_from_config(config,self.weights)
                if self.embedding_path is not None:
                    model.embedding_manager.load(self.embedding_path)
                self.model = model.to(self.device)
                # model.to doesn't change the cond_stage_model.device used to move the tokenizer output, so set it here
                self.model.cond_stage_model.device = self.device
            except AttributeError:
                raise SystemExit

            msg = f'setting sampler to {self.sampler_name}'
            if self.sampler_name=='plms':
                self.sampler = PLMSSampler(self.model, device=self.device)
            elif self.sampler_name == 'ddim':
                self.sampler = DDIMSampler(self.model, device=self.device)
            elif self.sampler_name == 'k_dpm_2_a':
                self.sampler = KSampler(self.model,'dpm_2_ancestral')
            elif self.sampler_name == 'k_dpm_2':
                self.sampler = KSampler(self.model,'dpm_2')
            elif self.sampler_name == 'k_euler_a':
                self.sampler = KSampler(self.model,'euler_ancestral')
            elif self.sampler_name == 'k_euler':
                self.sampler = KSampler(self.model,'euler')
            elif self.sampler_name == 'k_heun':
                self.sampler = KSampler(self.model,'heun')
            elif self.sampler_name == 'k_lms':
                self.sampler = KSampler(self.model,'lms')
            else:
                msg = f'unsupported sampler {self.sampler_name}, defaulting to plms'
                self.sampler = PLMSSampler(self.model, device=self.device)

            print(msg)

        return self.model
                
    def _load_model_from_config(self, config, ckpt):
        print(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cpu")
        if "global_step" in pl_sd:
            print(f"Global Step: {pl_sd['global_step']}")
        sd = pl_sd["state_dict"]
        model = instantiate_from_config(config.model)
        m, u = model.load_state_dict(sd, strict=False)
        model.eval()
        if self.full_precision:
            print('Using slower but more accurate full-precision math (--full_precision)')
        else:
            print('Using half precision math. Call with --full_precision to use slower but more accurate full precision.')
            model.half()
        return model

    def _load_img(self,path):
        image = Image.open(path).convert("RGB")
        w, h = image.size
        print(f"loaded input image of size ({w}, {h}) from {path}")
        w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
        image = image.resize((w, h), resample=Image.Resampling.LANCZOS)
        image = np.array(image).astype(np.float32) / 255.0
        image = image[None].transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)
        return 2.*image - 1.

    def _split_weighted_subprompts(text):
        """
        grabs all text up to the first occurrence of ':' 
        uses the grabbed text as a sub-prompt, and takes the value following ':' as weight
        if ':' has no value defined, defaults to 1.0
        repeats until no text remaining
        """
        remaining = len(text)
        prompts = []
        weights = []
        while remaining > 0:
            if ":" in text:
                idx = text.index(":") # first occurrence from start
                # grab up to index as sub-prompt
                prompt = text[:idx]
                remaining -= idx
                # remove from main text
                text = text[idx+1:]
                # find value for weight 
                if " " in text:
                    idx = text.index(" ") # first occurence
                else: # no space, read to end
                    idx = len(text)
                if idx != 0:
                    try:
                        weight = float(text[:idx])
                    except: # couldn't treat as float
                        print(f"Warning: '{text[:idx]}' is not a value, are you missing a space?")
                        weight = 1.0
                else: # no value found
                    weight = 1.0
                # remove from main text
                remaining -= idx
                text = text[idx+1:]
                # append the sub-prompt and its weight
                prompts.append(prompt)
                weights.append(weight)
            else: # no : found
                if len(text) > 0: # there is still text though
                    # take remainder as weight 1
                    prompts.append(text)
                    weights.append(1.0)
                remaining = 0
        return prompts, weights

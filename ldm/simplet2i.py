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
          sampler_name= ['ddim','plms','klms']  // klms
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

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim     import DDIMSampler
from ldm.models.diffusion.plms     import PLMSSampler
from ldm.models.diffusion.ksampler import KSampler

class T2I:
    """T2I class
    Attributes
    ----------
    outdir
    model
    config
    iterations
    batch_size
    steps
    seed
    sampler_name
    grid
    individual
    width
    height
    cfg_scale
    fixed_code
    latent_channels
    downsampling_factor
    precision
    strength

The vast majority of these arguments default to reasonable values.
"""
    def __init__(self,
                 outdir="outputs/txt2img-samples",
                 batch_size=1,
                 iterations = 1,
                 width=512,
                 height=512,
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
                 fixed_code=False,
                 precision='autocast',
                 full_precision=False,
                 strength=0.75, # default in scripts/img2img.py
                 latent_diffusion_weights=False  # just to keep track of this parameter when regenerating prompt
    ):
        self.outdir     = outdir
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
        self.fixed_code    = fixed_code
        self.latent_channels     = latent_channels
        self.downsampling_factor = downsampling_factor
        self.ddim_eta            = ddim_eta
        self.precision           = precision
        self.full_precision      = full_precision
        self.strength            = strength
        self.model      = None     # empty for now
        self.sampler    = None
        self.latent_diffusion_weights=latent_diffusion_weights
        if seed is None:
            self.seed = self._new_seed()
        else:
            self.seed = seed

    def txt2img(self,prompt,outdir=None,batch_size=None,iterations=None,
                steps=None,seed=None,grid=None,individual=None,width=None,height=None,
                cfg_scale=None,ddim_eta=None,strength=None,init_img=None):
        """
        Generate an image from the prompt, writing iteration images into the outdir
        The output is a list of lists in the format: [[filename1,seed1], [filename2,seed2],...]
        """
        outdir     = outdir     or self.outdir
        steps      = steps      or self.steps
        seed       = seed       or self.seed
        width      = width      or self.width
        height     = height     or self.height
        cfg_scale  = cfg_scale  or self.cfg_scale
        ddim_eta   = ddim_eta   or self.ddim_eta
        batch_size = batch_size or self.batch_size
        iterations = iterations or self.iterations
        strength   = strength   or self.strength     # not actually used here, but preserved for code refactoring

        model = self.load_model()  # will instantiate the model or return it from cache

        # grid and individual are mutually exclusive, with individual taking priority.
        # not necessary, but needed for compatability with dream bot
        if (grid is None):
            grid = self.grid
        if individual:
            grid = False
        
        data = [batch_size * [prompt]]

        # make directories and establish names for the output files
        os.makedirs(outdir, exist_ok=True)

        start_code = None
        if self.fixed_code:
            start_code = torch.randn([batch_size,
                                      self.latent_channels,
                                      height // self.downsampling_factor,
                                      width  // self.downsampling_factor],
                                     device=self.device)

        precision_scope = autocast if self.precision=="autocast" else nullcontext
        sampler         = self.sampler
        images = list()
        seeds  = list()
        filename = None
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
                                uc = model.get_learned_conditioning(batch_size * [""])
                            if isinstance(prompts, tuple):
                                prompts = list(prompts)
                            c = model.get_learned_conditioning(prompts)
                            shape = [self.latent_channels, height // self.downsampling_factor, width // self.downsampling_factor]
                            samples_ddim, _ = sampler.sample(S=steps,
                                                             conditioning=c,
                                                             batch_size=batch_size,
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
                                    filename = self._unique_filename(outdir,previousname=filename,
                                                                     seed=seed,isbatch=(batch_size>1))
                                    assert not os.path.exists(filename)
                                    Image.fromarray(x_sample.astype(np.uint8)).save(filename)
                                    images.append([filename,seed])
                            else:
                                all_samples.append(x_samples_ddim)
                                seeds.append(seed)

                        seed = self._new_seed()
 
                    if grid:
                        images = self._make_grid(samples=all_samples,
                                                 seeds=seeds,
                                                 batch_size=batch_size,
                                                 iterations=iterations,
                                                 outdir=outdir)

        toc = time.time()
        print(f'{batch_size * iterations} images generated in',"%4.2fs"% (toc-tic))

        return images
        
    # There is lots of shared code between this and txt2img and should be refactored.
    def img2img(self,prompt,outdir=None,init_img=None,batch_size=None,iterations=None,
                steps=None,seed=None,grid=None,individual=None,width=None,height=None,
                cfg_scale=None,ddim_eta=None,strength=None):
        """
        Generate an image from the prompt and the initial image, writing iteration images into the outdir
        The output is a list of lists in the format: [[filename1,seed1], [filename2,seed2],...]
        """
        outdir     = outdir     or self.outdir
        steps      = steps      or self.steps
        seed       = seed       or self.seed
        cfg_scale  = cfg_scale  or self.cfg_scale
        ddim_eta   = ddim_eta   or self.ddim_eta
        batch_size = batch_size or self.batch_size
        iterations = iterations or self.iterations
        strength   = strength   or self.strength

        if init_img is None:
            print("no init_img provided!")
            return []

        model = self.load_model()  # will instantiate the model or return it from cache

        precision_scope = autocast if self.precision=="autocast" else nullcontext

        # grid and individual are mutually exclusive, with individual taking priority.
        # not necessary, but needed for compatability with dream bot
        if (grid is None):
            grid = self.grid
        if individual:
            grid = False
        
        data = [batch_size * [prompt]]

        # PLMS sampler not supported yet, so ignore previous sampler
        if self.sampler_name!='ddim':
            print(f"sampler '{self.sampler_name}' is not yet supported. Using DDM sampler")
            sampler = DDIMSampler(model)
        else:
            sampler = self.sampler

        # make directories and establish names for the output files
        os.makedirs(outdir, exist_ok=True)

        assert os.path.isfile(init_img)
        init_image = self._load_img(init_img).to(self.device)
        init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
        with precision_scope("cuda"):
            init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))  # move to latent space

        sampler.make_schedule(ddim_num_steps=steps, ddim_eta=ddim_eta, verbose=False)

        try:
            assert 0. <= strength <= 1., 'can only work with strength in [0.0, 1.0]'
        except AssertionError:
            print(f"strength must be between 0.0 and 1.0, but received value {strength}")
            return []
        
        t_enc = int(strength * steps)
        print(f"target t_enc is {t_enc} steps")

        images = list()
        seeds  = list()
        filename = None
        
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
                                uc = model.get_learned_conditioning(batch_size * [""])
                            if isinstance(prompts, tuple):
                                prompts = list(prompts)
                            c = model.get_learned_conditioning(prompts)

                            # encode (scaled latent)
                            z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc]*batch_size).to(self.device))
                            # decode it
                            samples = sampler.decode(z_enc, c, t_enc, unconditional_guidance_scale=cfg_scale,
                                                     unconditional_conditioning=uc,)

                            x_samples = model.decode_first_stage(samples)
                            x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                            if not grid:
                                for x_sample in x_samples:
                                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                    filename = self._unique_filename(outdir,filename,seed=seed,isbatch=(batch_size>1))
                                    assert not os.path.exists(filename)
                                    Image.fromarray(x_sample.astype(np.uint8)).save(filename)
                                    images.append([filename,seed])
                            else:
                                all_samples.append(x_samples)
                                seeds.append(seed)

                        seed = self._new_seed()

                    if grid:
                        images = self._make_grid(samples=all_samples,
                                                 seeds=seeds,
                                                 batch_size=batch_size,
                                                 iterations=iterations,
                                                 outdir=outdir)

        toc = time.time()
        print(f'{batch_size * iterations} images generated in',"%4.2fs"% (toc-tic))

        return images

    def _make_grid(self,samples,seeds,batch_size,iterations,outdir):
        images = list()
        n_rows = batch_size if batch_size>1 else int(math.sqrt(batch_size * iterations))
        # save as grid
        grid = torch.stack(samples, 0)
        grid = rearrange(grid, 'n b c h w -> (n b) c h w')
        grid = make_grid(grid, nrow=n_rows)

        # to image
        grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
        filename = self._unique_filename(outdir,seed=seeds[0],grid_count=batch_size*iterations)
        Image.fromarray(grid.astype(np.uint8)).save(filename)
        for s in seeds:
            images.append([filename,s])
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
            elif self.sampler_name == 'klms':
                print("setting sampler to klms")
                self.sampler = KSampler(self.model,'lms')
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

    def _unique_filename(self,outdir,previousname=None,seed=0,isbatch=False,grid_count=None):
        revision = 1

        if previousname is None:
            # count up until we find an unfilled slot
            dir_list  = [a.split('.',1)[0] for a in os.listdir(outdir)]
            uniques   = dict.fromkeys(dir_list,True)
            basecount = 1
            while f'{basecount:06}' in uniques:
                basecount += 1
            if grid_count is not None:
                grid_label = f'grid#1-{grid_count}'
                filename = f'{basecount:06}.{seed}.{grid_label}.png'
            elif isbatch:
                filename = f'{basecount:06}.{seed}.01.png'
            else:
                filename = f'{basecount:06}.{seed}.png'
            
            return os.path.join(outdir,filename)

        else:
            previousname = os.path.basename(previousname)
            x = re.match('^(\d+)\..*\.png',previousname)
            if not x:
                return self._unique_filename(outdir,previousname,seed)

            basecount = int(x.groups()[0])
            series = 0 
            finished = False
            while not finished:
                series += 1
                filename = f'{basecount:06}.{seed}.png'
                if isbatch or os.path.exists(os.path.join(outdir,filename)):
                    filename = f'{basecount:06}.{seed}.{series:02}.png'
                finished = not os.path.exists(os.path.join(outdir,filename))
            return os.path.join(outdir,filename)

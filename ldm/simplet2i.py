# Copyright (c) 2022 Lincoln D. Stein (https://github.com/lstein)

# Derived from source code carrying the following copyrights
# Copyright (c) 2022 Machine Vision and Learning Group, LMU Munich
# Copyright (c) 2022 Robin Rombach and Patrick Esser and contributors

import torch
import numpy as np
import random
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
import transformers
import time
import re
import sys

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.ksampler import KSampler
from ldm.dream.pngwriter import PngWriter

"""Simplified text to image API for stable diffusion/latent diffusion

Example Usage:

from ldm.simplet2i import T2I

# Create an object with default values
t2i = T2I(model       = <path>        // models/ldm/stable-diffusion-v1/model.ckpt
          config      = <path>        // configs/stable-diffusion/v1-inference.yaml
          iterations  = <integer>     // how many times to run the sampling (1)
          batch_size  = <integer>     // how many images to generate per sampling (1)
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
# Will call load_model() if the model was not previously loaded and so
# may be slow at first.
# The method returns a list of images. Each row of the list is a sub-list of [filename,seed]
results = t2i.prompt2png(prompt     = "an astronaut riding a horse",
                         outdir     = "./outputs/samples",
                         iterations = 3)

for row in results:
    print(f'filename={row[0]}')
    print(f'seed    ={row[1]}')

# Same thing, but using an initial image.
results = t2i.prompt2png(prompt   = "an astronaut riding a horse",
                         outdir   = "./outputs/,
                         iterations = 3,
                         init_img = "./sketches/horse+rider.png")

for row in results:
    print(f'filename={row[0]}')
    print(f'seed    ={row[1]}')

# Same thing, but we return a series of Image objects, which lets you manipulate them,
# combine them, and save them under arbitrary names

results = t2i.prompt2image(prompt   = "an astronaut riding a horse"
                           outdir   = "./outputs/")
for row in results:
    im   = row[0]
    seed = row[1]
    im.save(f'./outputs/samples/an_astronaut_riding_a_horse-{seed}.png')
    im.thumbnail(100,100).save('./outputs/samples/astronaut_thumb.jpg')

Note that the old txt2img() and img2img() calls are deprecated but will
still work.
"""


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

    def __init__(
        self,
        batch_size=1,
        iterations=1,
        steps=50,
        seed=None,
        cfg_scale=7.5,
        weights='models/ldm/stable-diffusion-v1/model.ckpt',
        config='configs/stable-diffusion/v1-inference.yaml',
        width=512,
        height=512,
        sampler_name='klms',
        latent_channels=4,
        downsampling_factor=8,
        ddim_eta=0.0,  # deterministic
        precision='autocast',
        full_precision=False,
        strength=0.75,  # default in scripts/img2img.py
        embedding_path=None,
        # just to keep track of this parameter when regenerating prompt
        latent_diffusion_weights=False,
        device='cuda',
    ):
        self.batch_size = batch_size
        self.iterations = iterations
        self.width = width
        self.height = height
        self.steps = steps
        self.cfg_scale = cfg_scale
        self.weights = weights
        self.config = config
        self.sampler_name = sampler_name
        self.latent_channels = latent_channels
        self.downsampling_factor = downsampling_factor
        self.ddim_eta = ddim_eta
        self.precision = precision
        self.full_precision = full_precision
        self.strength = strength
        self.embedding_path = embedding_path
        self.model = None     # empty for now
        self.sampler = None
        self.latent_diffusion_weights = latent_diffusion_weights
        self.device = device

        self.session_peakmem = torch.cuda.max_memory_allocated()
        if seed is None:
            self.seed = self._new_seed()
        else:
            self.seed = seed
        transformers.logging.set_verbosity_error()

    def prompt2png(self, prompt, outdir, **kwargs):
        """
        Takes a prompt and an output directory, writes out the requested number
        of PNG files, and returns an array of [[filename,seed],[filename,seed]...]
        Optional named arguments are the same as those passed to T2I and prompt2image()
        """
        results = self.prompt2image(prompt, **kwargs)
        pngwriter = PngWriter(
            outdir, prompt, kwargs.get('batch_size', self.batch_size)
        )
        for r in results:
            pngwriter.write_image(r[0], r[1])
        return pngwriter.files_written

    def txt2img(self, prompt, **kwargs):
        outdir = kwargs.pop('outdir', 'outputs/img-samples')
        return self.prompt2png(prompt, outdir, **kwargs)

    def img2img(self, prompt, **kwargs):
        outdir = kwargs.pop('outdir', 'outputs/img-samples')
        assert (
            'init_img' in kwargs
        ), 'call to img2img() must include the init_img argument'
        return self.prompt2png(prompt, outdir, **kwargs)

    def prompt2image(
        self,
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
        gfpgan_strength=0,
        save_original=False,
        upscale=None,
        variants=None,
        sampler_name=None,
        **args,
    ):   # eat up additional cruft
        """
        ldm.prompt2image() is the common entry point for txt2img() and img2img()
        It takes the following arguments:
           prompt                          // prompt string (no default)
           iterations                      // iterations (1); image count=iterations x batch_size
           batch_size                      // images per iteration (1)
           steps                           // refinement steps per iteration
           seed                            // seed for random number generator
           width                           // width of image, in multiples of 64 (512)
           height                          // height of image, in multiples of 64 (512)
           cfg_scale                       // how strongly the prompt influences the image (7.5) (must be >1)
           init_img                        // path to an initial image - its dimensions override width and height
           strength                        // strength for noising/unnoising init_img. 0.0 preserves image exactly, 1.0 replaces it completely
           gfpgan_strength                 // strength for GFPGAN. 0.0 preserves image exactly, 1.0 replaces it completely
           ddim_eta                        // image randomness (eta=0.0 means the same seed always produces the same image)
           variants                        // if >0, the 1st generated image will be passed back to img2img to generate the requested number of variants
           image_callback                  // a function or method that will be called each time an image is generated

        To use the callback, define a function of method that receives two arguments, an Image object
        and the seed. You can then do whatever you like with the image, including converting it to
        different formats and manipulating it. For example:

            def process_image(image,seed):
                image.save(f{'images/seed.png'})

        The callback used by the prompt2png() can be found in ldm/dream_util.py. It contains code
        to create the requested output directory, select a unique informative name for each image, and
        write the prompt into the PNG metadata.
        """
        steps = steps or self.steps
        seed = seed or self.seed
        width = width or self.width
        height = height or self.height
        cfg_scale = cfg_scale or self.cfg_scale
        ddim_eta = ddim_eta or self.ddim_eta
        batch_size = batch_size or self.batch_size
        iterations = iterations or self.iterations
        strength = strength or self.strength

        model = (
            self.load_model()
        )  # will instantiate the model or return it from cache
        assert cfg_scale > 1.0, 'CFG_Scale (-C) must be >1.0'
        assert (
            0.0 <= strength <= 1.0
        ), 'can only work with strength in [0.0, 1.0]'
        w = int(width / 64) * 64
        h = int(height / 64) * 64
        if h != height or w != width:
            print(
                f'Height and width must be multiples of 64. Resizing to {h}x{w}.'
            )
            height = h
            width = w

        scope = autocast if self.precision == 'autocast' else nullcontext

        if sampler_name and (sampler_name != self.sampler_name):
            self.sampler_name = sampler_name
            self._set_sampler()

        tic = time.time()
        torch.cuda.torch.cuda.reset_peak_memory_stats()
        results = list()

        try:
            if init_img:
                assert os.path.exists(init_img), f'{init_img}: File not found'
                images_iterator = self._img2img(
                    prompt,
                    precision_scope=scope,
                    batch_size=batch_size,
                    steps=steps,
                    cfg_scale=cfg_scale,
                    ddim_eta=ddim_eta,
                    skip_normalize=skip_normalize,
                    init_img=init_img,
                    strength=strength,
                )
            else:
                images_iterator = self._txt2img(
                    prompt,
                    precision_scope=scope,
                    batch_size=batch_size,
                    steps=steps,
                    cfg_scale=cfg_scale,
                    ddim_eta=ddim_eta,
                    skip_normalize=skip_normalize,
                    width=width,
                    height=height,
                )

            with scope(self.device.type), self.model.ema_scope():
                for n in trange(iterations, desc='Generating'):
                    seed_everything(seed)
                    iter_images = next(images_iterator)
                    for image in iter_images:
                        results.append([image, seed])
                        if image_callback is not None:
                            image_callback(image, seed)
                    seed = self._new_seed()

                if upscale is not None or gfpgan_strength > 0:
                    for result in results:
                        image, seed = result
                        try:
                            if upscale is not None:
                                from ldm.gfpgan.gfpgan_tools import (
                                    real_esrgan_upscale,
                                )
                                if len(upscale) < 2:
                                    upscale.append(0.75)
                                image = real_esrgan_upscale(
                                    image,
                                    upscale[1],
                                    int(upscale[0]),
                                    prompt,
                                    seed,
                                )
                            if gfpgan_strength > 0:
                                from ldm.gfpgan.gfpgan_tools import _run_gfpgan

                                image = _run_gfpgan(
                                    image, gfpgan_strength, prompt, seed, 1
                                )
                        except Exception as e:
                            print(
                                f'Error running RealESRGAN - Your image was not upscaled.\n{e}'
                            )
                        if image_callback is not None:
                            if save_original:
                                image_callback(image, seed)
                            else:
                                image_callback(image, seed, upscaled=True)
                        else: # no callback passed, so we simply replace old image with rescaled one
                            result[0] = image

        except KeyboardInterrupt:
            print('*interrupted*')
            print(
                'Partial results will be returned; if --grid was requested, nothing will be returned.'
            )
        except RuntimeError as e:
            print(str(e))
            print('Are you sure your system has an adequate NVIDIA GPU?')

        toc = time.time()
        self.session_peakmem = max(
            self.session_peakmem, torch.cuda.max_memory_allocated()
        )
        print('Usage stats:')
        print(
            f'   {len(results)} image(s) generated in', '%4.2fs' % (toc - tic)
        )
        print(
            f'   Max VRAM used for this generation:',
            '%4.2fG' % (torch.cuda.max_memory_allocated() / 1e9),
        )
        print(
            f'   Max VRAM used since script start: ',
            '%4.2fG' % (self.session_peakmem / 1e9),
        )
        return results

    @torch.no_grad()
    def _txt2img(
        self,
        prompt,
        precision_scope,
        batch_size,
        steps,
        cfg_scale,
        ddim_eta,
        skip_normalize,
        width,
        height,
    ):
        """
        An infinite iterator of images from the prompt.
        """

        sampler = self.sampler

        while True:
            uc, c = self._get_uc_and_c(prompt, batch_size, skip_normalize)
            shape = [
                self.latent_channels,
                height // self.downsampling_factor,
                width // self.downsampling_factor,
            ]
            samples, _ = sampler.sample(
                S=steps,
                conditioning=c,
                batch_size=batch_size,
                shape=shape,
                verbose=False,
                unconditional_guidance_scale=cfg_scale,
                unconditional_conditioning=uc,
                eta=ddim_eta,
            )
            yield self._samples_to_images(samples)

    @torch.no_grad()
    def _img2img(
        self,
        prompt,
        precision_scope,
        batch_size,
        steps,
        cfg_scale,
        ddim_eta,
        skip_normalize,
        init_img,
        strength,
    ):
        """
        An infinite iterator of images from the prompt and the initial image
        """

        # PLMS sampler not supported yet, so ignore previous sampler
        if self.sampler_name != 'ddim':
            print(
                f"sampler '{self.sampler_name}' is not yet supported. Using DDM sampler"
            )
            sampler = DDIMSampler(self.model, device=self.device)
        else:
            sampler = self.sampler

        init_image = self._load_img(init_img).to(self.device)
        init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
        with precision_scope(self.device.type):
            init_latent = self.model.get_first_stage_encoding(
                self.model.encode_first_stage(init_image)
            )  # move to latent space

        sampler.make_schedule(
            ddim_num_steps=steps, ddim_eta=ddim_eta, verbose=False
        )

        t_enc = int(strength * steps)
        # print(f"target t_enc is {t_enc} steps")

        while True:
            uc, c = self._get_uc_and_c(prompt, batch_size, skip_normalize)

            # encode (scaled latent)
            z_enc = sampler.stochastic_encode(
                init_latent, torch.tensor([t_enc] * batch_size).to(self.device)
            )
            # decode it
            samples = sampler.decode(
                z_enc,
                c,
                t_enc,
                unconditional_guidance_scale=cfg_scale,
                unconditional_conditioning=uc,
            )
            yield self._samples_to_images(samples)

    # TODO: does this actually need to run every loop? does anything in it vary by random seed?
    def _get_uc_and_c(self, prompt, batch_size, skip_normalize):

        uc = self.model.get_learned_conditioning(batch_size * [''])

        # weighted sub-prompts
        subprompts, weights = T2I._split_weighted_subprompts(prompt)
        if len(subprompts) > 1:
            # i dont know if this is correct.. but it works
            c = torch.zeros_like(uc)
            # get total weight for normalizing
            totalWeight = sum(weights)
            # normalize each "sub prompt" and add it
            for i in range(0, len(subprompts)):
                weight = weights[i]
                if not skip_normalize:
                    weight = weight / totalWeight
                c = torch.add(
                    c,
                    self.model.get_learned_conditioning(
                        batch_size * [subprompts[i]]
                    ),
                    alpha=weight,
                )
        else:   # just standard 1 prompt
            c = self.model.get_learned_conditioning(batch_size * [prompt])
        return (uc, c)

    def _samples_to_images(self, samples):
        x_samples = self.model.decode_first_stage(samples)
        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
        images = list()
        for x_sample in x_samples:
            x_sample = 255.0 * rearrange(
                x_sample.cpu().numpy(), 'c h w -> h w c'
            )
            image = Image.fromarray(x_sample.astype(np.uint8))
            images.append(image)
        return images

    def _new_seed(self):
        self.seed = random.randrange(0, np.iinfo(np.uint32).max)
        return self.seed

    def _get_device(self):
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')

    def load_model(self):
        """Load and initialize the model from configuration variables passed at object creation time"""
        if self.model is None:
            seed_everything(self.seed)
            try:
                config = OmegaConf.load(self.config)
                self.device = self._get_device()
                model = self._load_model_from_config(config, self.weights)
                if self.embedding_path is not None:
                    model.embedding_manager.load(
                        self.embedding_path, self.full_precision
                    )
                self.model = model.to(self.device)
                # model.to doesn't change the cond_stage_model.device used to move the tokenizer output, so set it here
                self.model.cond_stage_model.device = self.device
            except AttributeError:
                import traceback
                print('Error loading model. Only the CUDA backend is supported',file=sys.stderr)
                print(traceback.format_exc(),file=sys.stderr)
                raise SystemExit

            self._set_sampler()

        return self.model

    def _set_sampler(self):
        msg = f'>> Setting Sampler to {self.sampler_name}'
        if self.sampler_name == 'plms':
            self.sampler = PLMSSampler(self.model, device=self.device)
        elif self.sampler_name == 'ddim':
            self.sampler = DDIMSampler(self.model, device=self.device)
        elif self.sampler_name == 'k_dpm_2_a':
            self.sampler = KSampler(
                self.model, 'dpm_2_ancestral', device=self.device
            )
        elif self.sampler_name == 'k_dpm_2':
            self.sampler = KSampler(self.model, 'dpm_2', device=self.device)
        elif self.sampler_name == 'k_euler_a':
            self.sampler = KSampler(
                self.model, 'euler_ancestral', device=self.device
            )
        elif self.sampler_name == 'k_euler':
            self.sampler = KSampler(self.model, 'euler', device=self.device)
        elif self.sampler_name == 'k_heun':
            self.sampler = KSampler(self.model, 'heun', device=self.device)
        elif self.sampler_name == 'k_lms':
            self.sampler = KSampler(self.model, 'lms', device=self.device)
        else:
            msg = f'>> Unsupported Sampler: {self.sampler_name}, Defaulting to plms'
            self.sampler = PLMSSampler(self.model, device=self.device)

        print(msg)

    def _load_model_from_config(self, config, ckpt):
        print(f'Loading model from {ckpt}')
        pl_sd = torch.load(ckpt, map_location='cpu')
        #        if "global_step" in pl_sd:
        #            print(f"Global Step: {pl_sd['global_step']}")
        sd = pl_sd['state_dict']
        model = instantiate_from_config(config.model)
        m, u = model.load_state_dict(sd, strict=False)
        model.to(self.device)
        model.eval()
        if self.full_precision:
            print(
                'Using slower but more accurate full-precision math (--full_precision)'
            )
        else:
            print(
                'Using half precision math. Call with --full_precision to use more accurate but VRAM-intensive full precision.'
            )
            model.half()
        return model

    def _load_img(self, path):
        print(f'image path = {path}, cwd = {os.getcwd()}')
        with Image.open(path) as img:
            image = img.convert('RGB')

        w, h = image.size
        print(f'loaded input image of size ({w}, {h}) from {path}')
        w, h = map(
            lambda x: x - x % 32, (w, h)
        )  # resize to integer multiple of 32
        image = image.resize((w, h), resample=Image.Resampling.LANCZOS)
        image = np.array(image).astype(np.float32) / 255.0
        image = image[None].transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)
        return 2.0 * image - 1.0

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
            if ':' in text:
                idx = text.index(':')   # first occurrence from start
                # grab up to index as sub-prompt
                prompt = text[:idx]
                remaining -= idx
                # remove from main text
                text = text[idx + 1 :]
                # find value for weight
                if ' ' in text:
                    idx = text.index(' ')   # first occurence
                else:   # no space, read to end
                    idx = len(text)
                if idx != 0:
                    try:
                        weight = float(text[:idx])
                    except:   # couldn't treat as float
                        print(
                            f"Warning: '{text[:idx]}' is not a value, are you missing a space?"
                        )
                        weight = 1.0
                else:   # no value found
                    weight = 1.0
                # remove from main text
                remaining -= idx
                text = text[idx + 1 :]
                # append the sub-prompt and its weight
                prompts.append(prompt)
                weights.append(weight)
            else:   # no : found
                if len(text) > 0:   # there is still text though
                    # take remainder as weight 1
                    prompts.append(text)
                    weights.append(1.0)
                remaining = 0
        return prompts, weights

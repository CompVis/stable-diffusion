# Copyright (c) 2022 Lincoln D. Stein (https://github.com/lstein)

# Derived from source code carrying the following copyrights
# Copyright (c) 2022 Machine Vision and Learning Group, LMU Munich
# Copyright (c) 2022 Robin Rombach and Patrick Esser and contributors

import torch
import numpy as np
import random
import os
import traceback
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

from ldm.util                      import instantiate_from_config
from ldm.models.diffusion.ddim     import DDIMSampler
from ldm.models.diffusion.plms     import PLMSSampler
from ldm.models.diffusion.ksampler import KSampler
from ldm.dream.pngwriter           import PngWriter
from ldm.dream.image_util          import InitImageResizer
from ldm.dream.devices import choose_autocast_device, choose_torch_device

"""Simplified text to image API for stable diffusion/latent diffusion

Example Usage:

from ldm.simplet2i import T2I

# Create an object with default values
t2i = T2I(model       = <path>        // models/ldm/stable-diffusion-v1/model.ckpt
          config      = <path>        // configs/stable-diffusion/v1-inference.yaml
          iterations  = <integer>     // how many times to run the sampling (1)
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
            iterations=1,
            steps=50,
            seed=None,
            cfg_scale=7.5,
            weights='models/ldm/stable-diffusion-v1/model.ckpt',
            config='configs/stable-diffusion/v1-inference.yaml',
            grid=False,
            width=512,
            height=512,
            sampler_name='k_lms',
            latent_channels=4,
            downsampling_factor=8,
            ddim_eta=0.0,  # deterministic
            precision='autocast',
            full_precision=False,
            strength=0.75,  # default in scripts/img2img.py
            embedding_path=None,
            device_type = 'cuda',
            # just to keep track of this parameter when regenerating prompt
            # needs to be replaced when new configuration system implemented.
            latent_diffusion_weights=False,
    ):
        self.iterations               = iterations
        self.width                    = width
        self.height                   = height
        self.steps                    = steps
        self.cfg_scale                = cfg_scale
        self.weights                  = weights
        self.config                   = config
        self.sampler_name             = sampler_name
        self.latent_channels          = latent_channels
        self.downsampling_factor      = downsampling_factor
        self.grid                     = grid
        self.ddim_eta                 = ddim_eta
        self.precision                = precision
        self.full_precision           = True if choose_torch_device() == 'mps' else full_precision
        self.strength                 = strength
        self.embedding_path           = embedding_path
        self.device_type              = device_type
        self.model                    = None     # empty for now
        self.sampler                  = None
        self.device                   = None
        self.latent_diffusion_weights = latent_diffusion_weights

        if device_type == 'cuda' and not torch.cuda.is_available():
            device_type = choose_torch_device()
            print(">> cuda not available, using device", device_type)
        self.device = torch.device(device_type)

        # for VRAM usage statistics
        device_type          = choose_torch_device()
        self.session_peakmem = torch.cuda.max_memory_allocated() if device_type == 'cuda' else None

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
        pngwriter = PngWriter(outdir)
        prefix = pngwriter.unique_prefix()
        outputs = []
        for image, seed in results:
            name = f'{prefix}.{seed}.png'
            path = pngwriter.save_image_and_prompt_to_png(
                image, f'{prompt} -S{seed}', name)
            outputs.append([path, seed])
        return outputs

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
            iterations     =    None,
            steps          =    None,
            seed           =    None,
            cfg_scale      =    None,
            ddim_eta       =    None,
            skip_normalize =    False,
            image_callback =    None,
            step_callback  =    None,
            width          =    None,
            height         =    None,
            # these are specific to img2img
            init_img       =    None,
            fit            =    False,
            strength       =    None,
            gfpgan_strength=    0,
            save_original  =    False,
            upscale        =    None,
            sampler_name   =    None,
            log_tokenization=  False,
            with_variations =   None,
            variation_amount =  0.0,
            **args,
    ):   # eat up additional cruft
        """
        ldm.prompt2image() is the common entry point for txt2img() and img2img()
        It takes the following arguments:
           prompt                          // prompt string (no default)
           iterations                      // iterations (1); image count=iterations
           steps                           // refinement steps per iteration
           seed                            // seed for random number generator
           width                           // width of image, in multiples of 64 (512)
           height                          // height of image, in multiples of 64 (512)
           cfg_scale                       // how strongly the prompt influences the image (7.5) (must be >1)
           init_img                        // path to an initial image - its dimensions override width and height
           strength                        // strength for noising/unnoising init_img. 0.0 preserves image exactly, 1.0 replaces it completely
           gfpgan_strength                 // strength for GFPGAN. 0.0 preserves image exactly, 1.0 replaces it completely
           ddim_eta                        // image randomness (eta=0.0 means the same seed always produces the same image)
           step_callback                   // a function or method that will be called each step
           image_callback                  // a function or method that will be called each time an image is generated
           with_variations                 // a weighted list [(seed_1, weight_1), (seed_2, weight_2), ...] of variations which should be applied before doing any generation
           variation_amount                // optional 0-1 value to slerp from -S noise to random noise (allows variations on an image)

        To use the step callback, define a function that receives two arguments:
        - Image GPU data
        - The step number

        To use the image callback, define a function of method that receives two arguments, an Image object
        and the seed. You can then do whatever you like with the image, including converting it to
        different formats and manipulating it. For example:

            def process_image(image,seed):
                image.save(f{'images/seed.png'})

        The callback used by the prompt2png() can be found in ldm/dream_util.py. It contains code
        to create the requested output directory, select a unique informative name for each image, and
        write the prompt into the PNG metadata.
        """
        # TODO: convert this into a getattr() loop
        steps                 = steps      or self.steps
        width                 = width      or self.width
        height                = height     or self.height
        cfg_scale             = cfg_scale  or self.cfg_scale
        ddim_eta              = ddim_eta   or self.ddim_eta
        iterations            = iterations or self.iterations
        strength              = strength   or self.strength
        self.log_tokenization = log_tokenization
        with_variations = [] if with_variations is None else with_variations

        model = (
            self.load_model()
        )  # will instantiate the model or return it from cache
        assert cfg_scale > 1.0, 'CFG_Scale (-C) must be >1.0'
        assert (
            0.0 <= strength <= 1.0
        ), 'can only work with strength in [0.0, 1.0]'
        assert (
                0.0 <= variation_amount <= 1.0
        ), '-v --variation_amount must be in [0.0, 1.0]'

        if len(with_variations) > 0 or variation_amount > 0.0:
            assert seed is not None,\
                'seed must be specified when using with_variations'
            if variation_amount == 0.0:
                assert iterations == 1,\
                    'when using --with_variations, multiple iterations are only possible when using --variation_amount'
            assert all(0 <= weight <= 1 for _, weight in with_variations),\
                f'variation weights must be in [0.0, 1.0]: got {[weight for _, weight in with_variations]}'

        seed                  = seed       or self.seed
        width, height, _ = self._resolution_check(width, height, log=True)

        # TODO: - Check if this is still necessary to run on M1 devices.
        #       - Move code into ldm.dream.devices to live alongside other
        #         special-hardware casing code.
        if self.precision == 'autocast' and torch.cuda.is_available():
            scope = autocast
        else:
            scope = nullcontext

        if sampler_name and (sampler_name != self.sampler_name):
            self.sampler_name = sampler_name
            self._set_sampler()

        tic = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        results = list()

        try:
            if init_img:
                assert os.path.exists(init_img), f'{init_img}: File not found'
                init_image = self._load_img(init_img, width, height, fit).to(self.device)
                with scope(self.device.type):
                    init_latent = self.model.get_first_stage_encoding(
                        self.model.encode_first_stage(init_image)
                    ) # move to latent space

                print(f' DEBUG: seed at make_image time ={seed}')
                make_image = self._img2img(
                    prompt,
                    steps=steps,
                    cfg_scale=cfg_scale,
                    ddim_eta=ddim_eta,
                    skip_normalize=skip_normalize,
                    init_latent=init_latent,
                    strength=strength,
                    callback=step_callback,
                )
            else:
                init_latent = None
                make_image = self._txt2img(
                    prompt,
                    steps=steps,
                    cfg_scale=cfg_scale,
                    ddim_eta=ddim_eta,
                    skip_normalize=skip_normalize,
                    width=width,
                    height=height,
                    callback=step_callback,
                )

            initial_noise = None
            if variation_amount > 0 or len(with_variations) > 0:
                # use fixed initial noise plus random noise per iteration
                seed_everything(seed)
                initial_noise = self._get_noise(init_latent,width,height)
                for v_seed, v_weight in with_variations:
                    seed = v_seed
                    seed_everything(seed)
                    next_noise = self._get_noise(init_latent,width,height)
                    initial_noise = self.slerp(v_weight, initial_noise, next_noise)
                if variation_amount > 0:
                    random.seed() # reset RNG to an actually random state, so we can get a random seed for variations
                    seed = random.randrange(0,np.iinfo(np.uint32).max)

            device_type = choose_autocast_device(self.device)
            with scope(device_type), self.model.ema_scope():
                for n in trange(iterations, desc='Generating'):
                    x_T = None
                    if variation_amount > 0:
                        seed_everything(seed)
                        target_noise = self._get_noise(init_latent,width,height)
                        x_T = self.slerp(variation_amount, initial_noise, target_noise)
                    elif initial_noise is not None:
                        # i.e. we specified particular variations
                        x_T = initial_noise
                    else:
                        seed_everything(seed)
                        if self.device.type == 'mps':
                            x_T = self._get_noise(init_latent,width,height)
                        # make_image will do the equivalent of get_noise itself
                    print(f' DEBUG: seed at make_image() invocation time ={seed}')
                    image = make_image(x_T)
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
                                f'>> Error running RealESRGAN - Your image was not upscaled.\n{e}'
                            )
                        if image_callback is not None:
                            if save_original:
                                image_callback(image, seed)
                            else:
                                image_callback(image, seed, upscaled=True)
                        else:  # no callback passed, so we simply replace old image with rescaled one
                            result[0] = image

        except KeyboardInterrupt:
            print('*interrupted*')
            print(
                '>> Partial results will be returned; if --grid was requested, nothing will be returned.'
            )
        except RuntimeError as e:
            print(traceback.format_exc(), file=sys.stderr)
            print('>> Are you sure your system has an adequate NVIDIA GPU?')

        toc = time.time()
        print('>> Usage stats:')
        print(
            f'>>   {len(results)} image(s) generated in', '%4.2fs' % (toc - tic)
        )
        print(
            f'>>   Max VRAM used for this generation:',
            '%4.2fG' % (torch.cuda.max_memory_allocated() / 1e9),
        )

        if self.session_peakmem:
            self.session_peakmem = max(
                self.session_peakmem, torch.cuda.max_memory_allocated()
            )
            print(
                f'>>   Max VRAM used since script start: ',
                '%4.2fG' % (self.session_peakmem / 1e9),
            )
        return results

    @torch.no_grad()
    def _txt2img(
        self,
        prompt,
        steps,
        cfg_scale,
        ddim_eta,
        skip_normalize,
        width,
        height,
        callback,
    ):
        """
        Returns a function returning an image derived from the prompt and the initial image
        Return value depends on the seed at the time you call it
        """

        sampler = self.sampler

        def make_image(x_T):
            uc, c = self._get_uc_and_c(prompt, skip_normalize)
            shape = [
                self.latent_channels,
                height // self.downsampling_factor,
                width // self.downsampling_factor,
            ]
            samples, _ = sampler.sample(
                batch_size=1,
                S=steps,
                x_T=x_T,
                conditioning=c,
                shape=shape,
                verbose=False,
                unconditional_guidance_scale=cfg_scale,
                unconditional_conditioning=uc,
                eta=ddim_eta,
                img_callback=callback
            )
            return self._sample_to_image(samples)
        return make_image

    @torch.no_grad()
    def _img2img(
            self,
            prompt,
            steps,
            cfg_scale,
            ddim_eta,
            skip_normalize,
            init_latent,
            strength,
            callback,  # Currently not implemented for img2img
    ):
        """
        Returns a function returning an image derived from the prompt and the initial image
        Return value depends on the seed at the time you call it
        """

        # PLMS sampler not supported yet, so ignore previous sampler
        if self.sampler_name != 'ddim':
            print(
                f">> sampler '{self.sampler_name}' is not yet supported. Using DDIM sampler"
            )
            sampler = DDIMSampler(self.model, device=self.device)
        else:
            sampler = self.sampler

        sampler.make_schedule(
            ddim_num_steps=steps, ddim_eta=ddim_eta, verbose=False
        )

        t_enc = int(strength * steps)

        def make_image(x_T):
            uc, c = self._get_uc_and_c(prompt, skip_normalize)

            # encode (scaled latent)
            z_enc = sampler.stochastic_encode(
                init_latent,
                torch.tensor([t_enc]).to(self.device),
                noise=x_T
            )
            # decode it
            samples = sampler.decode(
                z_enc,
                c,
                t_enc,
                img_callback=callback,
                unconditional_guidance_scale=cfg_scale,
                unconditional_conditioning=uc,
            )
            return self._sample_to_image(samples)
        return make_image

    # TODO: does this actually need to run every loop? does anything in it vary by random seed?
    def _get_uc_and_c(self, prompt, skip_normalize):

        uc = self.model.get_learned_conditioning([''])

        # get weighted sub-prompts
        weighted_subprompts = T2I._split_weighted_subprompts(
            prompt, skip_normalize)

        if len(weighted_subprompts) > 1:
            # i dont know if this is correct.. but it works
            c = torch.zeros_like(uc)
            # normalize each "sub prompt" and add it
            for subprompt, weight in weighted_subprompts:
                self._log_tokenization(subprompt)
                c = torch.add(
                    c,
                    self.model.get_learned_conditioning([subprompt]),
                    alpha=weight,
                )
        else:   # just standard 1 prompt
            self._log_tokenization(prompt)
            c = self.model.get_learned_conditioning([prompt])
        return (uc, c)

    def _sample_to_image(self, samples):
        x_samples = self.model.decode_first_stage(samples)
        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
        if len(x_samples) != 1:
            raise Exception(
                f'>> expected to get a single image, but got {len(x_samples)}')
        x_sample = 255.0 * rearrange(
            x_samples[0].cpu().numpy(), 'c h w -> h w c'
        )
        return Image.fromarray(x_sample.astype(np.uint8))

    def _new_seed(self):
        self.seed = random.randrange(0, np.iinfo(np.uint32).max)
        return self.seed

    def load_model(self):
        """Load and initialize the model from configuration variables passed at object creation time"""
        if self.model is None:
            seed_everything(self.seed)
            try:
                config = OmegaConf.load(self.config)
                model = self._load_model_from_config(config, self.weights)
                if self.embedding_path is not None:
                    model.embedding_manager.load(
                        self.embedding_path, self.full_precision
                    )
                self.model = model.to(self.device)
                # model.to doesn't change the cond_stage_model.device used to move the tokenizer output, so set it here
                self.model.cond_stage_model.device = self.device
            except AttributeError as e:
                print(f'>> Error loading model. {str(e)}', file=sys.stderr)
                print(traceback.format_exc(), file=sys.stderr)
                raise SystemExit from e

            self._set_sampler()

        return self.model

    # returns a tensor filled with random numbers from a normal distribution
    def _get_noise(self,init_latent,width,height):
        if init_latent is not None:
            if self.device.type == 'mps':
                return torch.randn_like(init_latent, device='cpu').to(self.device)
            else:
                return torch.randn_like(init_latent, device=self.device)
        else:
            if self.device.type == 'mps':
                return torch.randn([1,
                                    self.latent_channels,
                                    height // self.downsampling_factor,
                                    width  // self.downsampling_factor],
                                   device='cpu').to(self.device)
            else:
                return torch.randn([1,
                                    self.latent_channels,
                                    height // self.downsampling_factor,
                                    width  // self.downsampling_factor],
                                   device=self.device)

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
        print(f'>> Loading model from {ckpt}')
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
                '>> Using half precision math. Call with --full_precision to use more accurate but VRAM-intensive full precision.'
            )
            model.half()
        return model

    def _load_img(self, path, width, height, fit=False):
        with Image.open(path) as img:
            image = img.convert('RGB')
        print(
            f'>> loaded input image of size {image.width}x{image.height} from {path}'
        )

        # The logic here is:
        # 1. If "fit" is true, then the image will be fit into the bounding box defined
        #    by width and height. It will do this in a way that preserves the init image's
        #    aspect ratio while preventing letterboxing. This means that if there is
        #    leftover horizontal space after rescaling the image to fit in the bounding box,
        #    the generated image's width will be reduced to the rescaled init image's width.
        #    Similarly for the vertical space.
        # 2. Otherwise, if "fit" is false, then the image will be scaled, preserving its
        #    aspect ratio, to the nearest multiple of 64. Large images may generate an
        #    unexpected OOM error.
        if fit:
            image = self._fit_image(image,(width,height))
        else:
            image = self._squeeze_image(image)
        image = np.array(image).astype(np.float32) / 255.0
        image = image[None].transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)
        return 2.0 * image - 1.0

    def _squeeze_image(self,image):
        x,y,resize_needed = self._resolution_check(image.width,image.height)
        if resize_needed:
            return InitImageResizer(image).resize(x,y)
        return image


    def _fit_image(self,image,max_dimensions):
        w,h = max_dimensions
        print(
            f'>> image will be resized to fit inside a box {w}x{h} in size.'
        )
        if image.width > image.height:
            h   = None   # by setting h to none, we tell InitImageResizer to fit into the width and calculate height
        elif image.height > image.width:
            w   = None   # ditto for w
        else:
            pass
        image = InitImageResizer(image).resize(w,h)   # note that InitImageResizer does the multiple of 64 truncation internally
        print(
            f'>> after adjusting image dimensions to be multiples of 64, init image is {image.width}x{image.height}'
            )
        return image


    # TO DO: Move this and related weighted subprompt code into its own module.
    def _split_weighted_subprompts(text, skip_normalize=False):
        """
        grabs all text up to the first occurrence of ':'
        uses the grabbed text as a sub-prompt, and takes the value following ':' as weight
        if ':' has no value defined, defaults to 1.0
        repeats until no text remaining
        """
        prompt_parser = re.compile("""
            (?P<prompt>     # capture group for 'prompt'
            (?:\\\:|[^:])+  # match one or more non ':' characters or escaped colons '\:'
            )               # end 'prompt'
            (?:             # non-capture group
            :+              # match one or more ':' characters
            (?P<weight>     # capture group for 'weight'
            -?\d+(?:\.\d+)? # match positive or negative integer or decimal number
            )?              # end weight capture group, make optional
            \s*             # strip spaces after weight
            |               # OR
            $               # else, if no ':' then match end of line
            )               # end non-capture group
        """, re.VERBOSE)
        parsed_prompts = [(match.group("prompt").replace("\\:", ":"), float(
            match.group("weight") or 1)) for match in re.finditer(prompt_parser, text)]
        if skip_normalize:
            return parsed_prompts
        weight_sum = sum(map(lambda x: x[1], parsed_prompts))
        if weight_sum == 0:
            print(
                "Warning: Subprompt weights add up to zero. Discarding and using even weights instead.")
            equal_weight = 1 / len(parsed_prompts)
            return [(x[0], equal_weight) for x in parsed_prompts]
        return [(x[0], x[1] / weight_sum) for x in parsed_prompts]

    # shows how the prompt is tokenized
    # usually tokens have '</w>' to indicate end-of-word,
    # but for readability it has been replaced with ' '
    def _log_tokenization(self, text):
        if not self.log_tokenization:
            return
        tokens = self.model.cond_stage_model.tokenizer._tokenize(text)
        tokenized = ""
        discarded = ""
        usedTokens = 0
        totalTokens = len(tokens)
        for i in range(0, totalTokens):
            token = tokens[i].replace('</w>', ' ')
            # alternate color
            s = (usedTokens % 6) + 1
            if i < self.model.cond_stage_model.max_length:
                tokenized = tokenized + f"\x1b[0;3{s};40m{token}"
                usedTokens += 1
            else:  # over max token length
                discarded = discarded + f"\x1b[0;3{s};40m{token}"
        print(f"\nTokens ({usedTokens}):\n{tokenized}\x1b[0m")
        if discarded != "":
            print(
                f"Tokens Discarded ({totalTokens-usedTokens}):\n{discarded}\x1b[0m")

    def _resolution_check(self, width, height, log=False):
        resize_needed = False
        w, h = map(
            lambda x: x - x % 64, (width, height)
        )  # resize to integer multiple of 64
        if h != height or w != width:
            if log:
                print(
                    f'>> Provided width and height must be multiples of 64. Auto-resizing to {w}x{h}'
                )
            height = h
            width  = w
            resize_needed = True

        if (width * height) > (self.width * self.height):
            print(">> This input is larger than your defaults. If you run out of memory, please use a smaller image.")

        return width, height, resize_needed


    def slerp(self, t, v0, v1, DOT_THRESHOLD=0.9995):
        '''
        Spherical linear interpolation
        Args:
            t (float/np.ndarray): Float value between 0.0 and 1.0
            v0 (np.ndarray): Starting vector
            v1 (np.ndarray): Final vector
            DOT_THRESHOLD (float): Threshold for considering the two vectors as
                                colineal. Not recommended to alter this.
        Returns:
            v2 (np.ndarray): Interpolation vector between v0 and v1
        '''
        inputs_are_torch = False
        if not isinstance(v0, np.ndarray):
            inputs_are_torch = True
            v0 = v0.detach().cpu().numpy()
        if not isinstance(v1, np.ndarray):
            inputs_are_torch = True
            v1 = v1.detach().cpu().numpy()

        dot = np.sum(v0 * v1 / (np.linalg.norm(v0) * np.linalg.norm(v1)))
        if np.abs(dot) > DOT_THRESHOLD:
            v2 = (1 - t) * v0 + t * v1
        else:
            theta_0 = np.arccos(dot)
            sin_theta_0 = np.sin(theta_0)
            theta_t = theta_0 * t
            sin_theta_t = np.sin(theta_t)
            s0 = np.sin(theta_0 - theta_t) / sin_theta_0
            s1 = sin_theta_t / sin_theta_0
            v2 = s0 * v0 + s1 * v1

        if inputs_are_torch:
            v2 = torch.from_numpy(v2).to(self.device)

        return v2

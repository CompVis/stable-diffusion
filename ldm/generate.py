# Copyright (c) 2022 Lincoln D. Stein (https://github.com/lstein)

# Derived from source code carrying the following copyrights
# Copyright (c) 2022 Machine Vision and Learning Group, LMU Munich
# Copyright (c) 2022 Robin Rombach and Patrick Esser and contributors

import torch
import numpy as np
import random
import os
import time
import re
import sys
import traceback
import transformers

from omegaconf import OmegaConf
from PIL import Image, ImageOps
from torch import nn
from pytorch_lightning import seed_everything

from ldm.util                      import instantiate_from_config
from ldm.models.diffusion.ddim     import DDIMSampler
from ldm.models.diffusion.plms     import PLMSSampler
from ldm.models.diffusion.ksampler import KSampler
from ldm.dream.pngwriter           import PngWriter
from ldm.dream.image_util          import InitImageResizer
from ldm.dream.devices             import choose_torch_device
from ldm.dream.conditioning        import get_uc_and_c

"""Simplified text to image API for stable diffusion/latent diffusion

Example Usage:

from ldm.generate import Generate

# Create an object with default values
gr = Generate()

# do the slow model initialization
gr.load_model()

# Do the fast inference & image generation. Any options passed here
# override the default values assigned during class initialization
# Will call load_model() if the model was not previously loaded and so
# may be slow at first.
# The method returns a list of images. Each row of the list is a sub-list of [filename,seed]
results = gr.prompt2png(prompt     = "an astronaut riding a horse",
                         outdir     = "./outputs/samples",
                         iterations = 3)

for row in results:
    print(f'filename={row[0]}')
    print(f'seed    ={row[1]}')

# Same thing, but using an initial image.
results = gr.prompt2png(prompt   = "an astronaut riding a horse",
                         outdir   = "./outputs/,
                         iterations = 3,
                         init_img = "./sketches/horse+rider.png")

for row in results:
    print(f'filename={row[0]}')
    print(f'seed    ={row[1]}')

# Same thing, but we return a series of Image objects, which lets you manipulate them,
# combine them, and save them under arbitrary names

results = gr.prompt2image(prompt   = "an astronaut riding a horse"
                           outdir   = "./outputs/")
for row in results:
    im   = row[0]
    seed = row[1]
    im.save(f'./outputs/samples/an_astronaut_riding_a_horse-{seed}.png')
    im.thumbnail(100,100).save('./outputs/samples/astronaut_thumb.jpg')

Note that the old txt2img() and img2img() calls are deprecated but will
still work.

The full list of arguments to Generate() are:
gr = Generate(
          weights     = path to model weights ('models/ldm/stable-diffusion-v1/model.ckpt')
          config     = path to model configuraiton ('configs/stable-diffusion/v1-inference.yaml')
          iterations  = <integer>     // how many times to run the sampling (1)
          steps       = <integer>     // 50
          seed        = <integer>     // current system time
          sampler_name= ['ddim', 'k_dpm_2_a', 'k_dpm_2', 'k_euler_a', 'k_euler', 'k_heun', 'k_lms', 'plms']  // k_lms
          grid        = <boolean>     // false
          width       = <integer>     // image width, multiple of 64 (512)
          height      = <integer>     // image height, multiple of 64 (512)
          cfg_scale   = <float>       // condition-free guidance scale (7.5)
          )

"""


class Generate:
    """Generate class
    Stores default values for multiple configuration items
    """

    def __init__(
            self,
            iterations            = 1,
            steps                 = 50,
            cfg_scale             = 7.5,
            weights               = 'models/ldm/stable-diffusion-v1/model.ckpt',
            config                = 'configs/stable-diffusion/v1-inference.yaml',
            grid                  = False,
            width                 = 512,
            height                = 512,
            sampler_name          = 'k_lms',
            ddim_eta              = 0.0,  # deterministic
            precision             = 'autocast',
            full_precision        = False,
            strength              = 0.75,  # default in scripts/img2img.py
            seamless              = False,
            embedding_path        = None,
            device_type           = 'cuda',
            ignore_ctrl_c         = False,
    ):
        self.iterations               = iterations
        self.width                    = width
        self.height                   = height
        self.steps                    = steps
        self.cfg_scale                = cfg_scale
        self.weights                  = weights
        self.config                   = config
        self.sampler_name             = sampler_name
        self.grid                     = grid
        self.ddim_eta                 = ddim_eta
        self.precision                = precision
        self.full_precision           = True if choose_torch_device() == 'mps' else full_precision
        self.strength                 = strength
        self.seamless                 = seamless
        self.embedding_path           = embedding_path
        self.device_type              = device_type
        self.ignore_ctrl_c            = ignore_ctrl_c    # note, this logic probably doesn't belong here...
        self.model                    = None     # empty for now
        self.sampler                  = None
        self.device                   = None
        self.generators               = {}
        self.base_generator           = None
        self.seed                     = None

        if device_type == 'cuda' and not torch.cuda.is_available():
            device_type = choose_torch_device()
            print(">> cuda not available, using device", device_type)
        self.device = torch.device(device_type)

        # for VRAM usage statistics
        device_type          = choose_torch_device()
        self.session_peakmem = torch.cuda.max_memory_allocated() if device_type == 'cuda' else None
        transformers.logging.set_verbosity_error()

    def prompt2png(self, prompt, outdir, **kwargs):
        """
        Takes a prompt and an output directory, writes out the requested number
        of PNG files, and returns an array of [[filename,seed],[filename,seed]...]
        Optional named arguments are the same as those passed to Generate and prompt2image()
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
            sampler_name   =    None,
            seamless       =    False,
            log_tokenization=  False,
            with_variations =   None,
            variation_amount =  0.0,
            # these are specific to img2img and inpaint
            init_img       =    None,
            init_mask      =    None,
            fit            =    False,
            strength       =    None,
            # these are specific to GFPGAN/ESRGAN
            gfpgan_strength=    0,
            save_original  =    False,
            upscale        =    None,
            **args,
    ):   # eat up additional cruft
        """
        ldm.generate.prompt2image() is the common entry point for txt2img() and img2img()
        It takes the following arguments:
           prompt                          // prompt string (no default)
           iterations                      // iterations (1); image count=iterations
           steps                           // refinement steps per iteration
           seed                            // seed for random number generator
           width                           // width of image, in multiples of 64 (512)
           height                          // height of image, in multiples of 64 (512)
           cfg_scale                       // how strongly the prompt influences the image (7.5) (must be >1)
           seamless                        // whether the generated image should tile
           init_img                        // path to an initial image
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
        seamless              = seamless   or self.seamless
        cfg_scale             = cfg_scale  or self.cfg_scale
        ddim_eta              = ddim_eta   or self.ddim_eta
        iterations            = iterations or self.iterations
        strength              = strength   or self.strength
        self.seed             = seed
        self.log_tokenization = log_tokenization
        with_variations = [] if with_variations is None else with_variations

        model = (
            self.load_model()
        )  # will instantiate the model or return it from cache

        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                m.padding_mode = 'circular' if seamless else m._orig_padding_mode
        
        assert cfg_scale > 1.0, 'CFG_Scale (-C) must be >1.0'
        assert (
            0.0 < strength < 1.0
        ), 'img2img and inpaint strength can only work with 0.0 < strength < 1.0'
        assert (
                0.0 <= variation_amount <= 1.0
        ), '-v --variation_amount must be in [0.0, 1.0]'

        # check this logic - doesn't look right
        if len(with_variations) > 0 or variation_amount > 1.0:
            assert seed is not None,\
                'seed must be specified when using with_variations'
            if variation_amount == 0.0:
                assert iterations == 1,\
                    'when using --with_variations, multiple iterations are only possible when using --variation_amount'
            assert all(0 <= weight <= 1 for _, weight in with_variations),\
                f'variation weights must be in [0.0, 1.0]: got {[weight for _, weight in with_variations]}'

        width, height, _ = self._resolution_check(width, height, log=True)

        if sampler_name and (sampler_name != self.sampler_name):
            self.sampler_name = sampler_name
            self._set_sampler()

        tic = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        results          = list()
        init_image       = None
        mask_image       = None

        try:
            uc, c = get_uc_and_c(
                prompt, model=self.model,
                skip_normalize=skip_normalize,
                log_tokens=self.log_tokenization
            )

            (init_image,mask_image) = self._make_images(init_img,init_mask, width, height, fit)
            
            if (init_image is not None) and (mask_image is not None):
                generator = self._make_inpaint()
            elif init_image is not None:
                generator = self._make_img2img()
            else:
                generator = self._make_txt2img()

            generator.set_variation(self.seed, variation_amount, with_variations)
            results = generator.generate(
                prompt,
                iterations     = iterations,
                seed           = self.seed,
                sampler        = self.sampler,
                steps          = steps,
                cfg_scale      = cfg_scale,
                conditioning   = (uc,c),
                ddim_eta       = ddim_eta,
                image_callback = image_callback,  # called after the final image is generated
                step_callback  = step_callback,   # called after each intermediate image is generated
                width          = width,
                height         = height,
                init_image     = init_image,      # notice that init_image is different from init_img
                mask_image     = mask_image,
                strength       = strength,
            )

            if upscale is not None or gfpgan_strength > 0:
                self.upscale_and_reconstruct(results,
                                             upscale        = upscale,
                                             strength       = gfpgan_strength,
                                             save_original  = save_original,
                                             image_callback = image_callback)

        except KeyboardInterrupt:
            print('*interrupted*')
            if not self.ignore_ctrl_c:
                raise KeyboardInterrupt
            print(
                '>> Partial results will be returned; if --grid was requested, nothing will be returned.'
            )
        except RuntimeError as e:
            print(traceback.format_exc(), file=sys.stderr)
            print('>> Could not generate image.')

        toc = time.time()
        print('>> Usage stats:')
        print(
            f'>>   {len(results)} image(s) generated in', '%4.2fs' % (toc - tic)
        )
        if torch.cuda.is_available() and self.device.type == 'cuda':
            print(
                f'>>   Max VRAM used for this generation:',
                '%4.2fG.' % (torch.cuda.max_memory_allocated() / 1e9),
                'Current VRAM utilization:'
                '%4.2fG' % (torch.cuda.memory_allocated() / 1e9),
            )

            self.session_peakmem = max(
                self.session_peakmem, torch.cuda.max_memory_allocated()
            )
            print(
                f'>>   Max VRAM used since script start: ',
                '%4.2fG' % (self.session_peakmem / 1e9),
            )
        return results

    def _make_images(self, img_path, mask_path, width, height, fit=False):
        init_image      = None
        init_mask       = None
        if not img_path:
            return None,None

        image        = self._load_img(img_path, width, height, fit=fit) # this returns an Image
        init_image   = self._create_init_image(image)                   # this returns a torch tensor

        if self._has_transparency(image) and not mask_path:      # if image has a transparent area and no mask was provided, then try to generate mask
            print('>> Initial image has transparent areas. Will inpaint in these regions.')
            if self._check_for_erasure(image):
                print(
                    '>> WARNING: Colors underneath the transparent region seem to have been erased.\n',
                    '>>          Inpainting will be suboptimal. Please preserve the colors when making\n',
                    '>>          a transparency mask, or provide mask explicitly using --init_mask (-M).'
                )
            init_mask = self._create_init_mask(image)                   # this returns a torch tensor

        if mask_path:
            mask_image  = self._load_img(mask_path, width, height, fit=fit) # this returns an Image
            init_mask   = self._create_init_mask(mask_image)

        return init_image,init_mask

    def _make_img2img(self):
        if not self.generators.get('img2img'):
            from ldm.dream.generator.img2img import Img2Img
            self.generators['img2img'] = Img2Img(self.model)
        return self.generators['img2img']

    def _make_txt2img(self):
        if not self.generators.get('txt2img'):
            from ldm.dream.generator.txt2img import Txt2Img
            self.generators['txt2img'] = Txt2Img(self.model)
        return self.generators['txt2img']

    def _make_inpaint(self):
        if not self.generators.get('inpaint'):
            from ldm.dream.generator.inpaint import Inpaint
            self.generators['inpaint'] = Inpaint(self.model)
        return self.generators['inpaint']

    def load_model(self):
        """Load and initialize the model from configuration variables passed at object creation time"""
        if self.model is None:
            seed_everything(random.randrange(0, np.iinfo(np.uint32).max))
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

            for m in self.model.modules():
                if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                    m._orig_padding_mode = m.padding_mode

        return self.model

    def upscale_and_reconstruct(self,
                                image_list,
                                upscale       = None,
                                strength      =  0.0,
                                save_original = False,
                                image_callback = None):
        try:
            if upscale is not None:
                from ldm.gfpgan.gfpgan_tools import real_esrgan_upscale
            if strength > 0:
                from ldm.gfpgan.gfpgan_tools import run_gfpgan
        except (ModuleNotFoundError, ImportError):
            print(traceback.format_exc(), file=sys.stderr)
            print('>> You may need to install the ESRGAN and/or GFPGAN modules')
            return
            
        for r in image_list:
            image, seed = r
            try:
                if upscale is not None:
                    if len(upscale) < 2:
                        upscale.append(0.75)
                    image = real_esrgan_upscale(
                        image,
                        upscale[1],
                        int(upscale[0]),
                        seed,
                    )
                if strength > 0:
                    image = run_gfpgan(
                        image, strength, seed, 1
                    )
            except Exception as e:
                print(
                    f'>> Error running RealESRGAN or GFPGAN. Your image was not upscaled.\n{e}'
                )

            if image_callback is not None:
                image_callback(image, seed, upscaled=True)
            else:
                r[0] = image

    # to help WebGUI - front end to generator util function
    def sample_to_image(self,samples):
        return self._sample_to_image(samples)

    def _sample_to_image(self,samples):
        if not self.base_generator:
            from ldm.dream.generator import Generator
            self.base_generator = Generator(self.model)
        return self.base_generator.sample_to_image(samples)

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

        # for usage statistics
        device_type = choose_torch_device()
        if device_type == 'cuda':
            torch.cuda.reset_peak_memory_stats() 
        tic = time.time()

        # this does the work
        pl_sd = torch.load(ckpt, map_location='cpu')
        sd = pl_sd['state_dict']
        model = instantiate_from_config(config.model)
        m, u = model.load_state_dict(sd, strict=False)
        
        if self.full_precision:
            print(
                '>> Using slower but more accurate full-precision math (--full_precision)'
            )
        else:
            print(
                '>> Using half precision math. Call with --full_precision to use more accurate but VRAM-intensive full precision.'
            )
            model.half()
        model.to(self.device)
        model.eval()

        # usage statistics
        toc = time.time()
        print(
            f'>> Model loaded in', '%4.2fs' % (toc - tic)
        )
        if device_type == 'cuda':
            print(
                '>> Max VRAM used to load the model:',
                '%4.2fG' % (torch.cuda.max_memory_allocated() / 1e9),
                '\n>> Current VRAM usage:'
                '%4.2fG' % (torch.cuda.memory_allocated() / 1e9),
            )

        return model

    def _load_img(self, path, width, height, fit=False):
        assert os.path.exists(path), f'>> {path}: File not found'

        #        with Image.open(path) as img:
        #            image = img.convert('RGBA')
        image = Image.open(path)
        print(
            f'>> loaded input image of size {image.width}x{image.height} from {path}'
        )
        if fit:
            image = self._fit_image(image,(width,height))
        else:
            image = self._squeeze_image(image)
        return image

    def _create_init_image(self,image):
        image = image.convert('RGB')
        # print(
        #     f'>> DEBUG: writing the image to img.png'
        # )
        # image.save('img.png')
        image = np.array(image).astype(np.float32) / 255.0
        image = image[None].transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)
        image = 2.0 * image - 1.0 
        return image.to(self.device)

    def _create_init_mask(self, image):
        # convert into a black/white mask
        image = self._image_to_mask(image)
        image = image.convert('RGB')
        # BUG: We need to use the model's downsample factor rather than hardcoding "8"
        from ldm.dream.generator.base import downsampling
        image = image.resize((image.width//downsampling, image.height//downsampling), resample=Image.Resampling.LANCZOS)
        # print(
        #     f'>> DEBUG: writing the mask to mask.png'
        #     )
        # image.save('mask.png')
        image = np.array(image)
        image = image.astype(np.float32) / 255.0
        image = image[None].transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)
        return image.to(self.device)

    # The mask is expected to have the region to be inpainted
    # with alpha transparency. It converts it into a black/white
    # image with the transparent part black.
    def _image_to_mask(self, mask_image, invert=False) -> Image:
        # Obtain the mask from the transparency channel
        mask = Image.new(mode="L", size=mask_image.size, color=255)
        mask.putdata(mask_image.getdata(band=3))
        if invert:
            mask = ImageOps.invert(mask)
        return mask

    def _has_transparency(self,image):
        if image.info.get("transparency", None) is not None:
            return True
        if image.mode == "P":
            transparent = image.info.get("transparency", -1)
            for _, index in image.getcolors():
                if index == transparent:
                    return True
        elif image.mode == "RGBA":
            extrema = image.getextrema()
            if extrema[3][0] < 255:
                return True
        return False

    
    def _check_for_erasure(self,image):
        width, height = image.size
        pixdata       = image.load()
        colored       = 0
        for y in range(height):
            for x in range(width):
                if pixdata[x, y][3] == 0:
                    r, g, b, _ = pixdata[x, y]
                    if (r, g, b) != (0, 0, 0) and \
                       (r, g, b) != (255, 255, 255):
                        colored += 1
        return colored == 0

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



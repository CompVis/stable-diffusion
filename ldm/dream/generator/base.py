'''
Base class for ldm.dream.generator.*
including img2img, txt2img, and inpaint
'''
import torch
import numpy as  np
import random
from tqdm import tqdm, trange
from PIL               import Image
from einops import rearrange, repeat
from pytorch_lightning import seed_everything
from ldm.dream.devices import choose_autocast_device

downsampling = 8

class Generator():
    def __init__(self,model):
        self.model               = model
        self.seed                = None
        self.latent_channels     = model.channels
        self.downsampling_factor = downsampling   # BUG: should come from model or config
        self.variation_amount    = 0
        self.with_variations     = []

    # this is going to be overridden in img2img.py, txt2img.py and inpaint.py
    def get_make_image(self,prompt,**kwargs):
        """
        Returns a function returning an image derived from the prompt and the initial image
        Return value depends on the seed at the time you call it
        """
        raise NotImplementedError("image_iterator() must be implemented in a descendent class")

    def set_variation(self, seed, variation_amount, with_variations):
        self.seed             = seed
        self.variation_amount = variation_amount
        self.with_variations  = with_variations

    def generate(self,prompt,init_image,width,height,iterations=1,seed=None,
                 image_callback=None, step_callback=None,
                 **kwargs):
        device_type,scope   = choose_autocast_device(self.model.device)
        make_image          = self.get_make_image(
            prompt,
            init_image    = init_image,
            width         = width,
            height        = height,
            step_callback = step_callback,
            **kwargs
        )

        results             = []
        seed                = seed if seed else self.new_seed()
        seed, initial_noise = self.generate_initial_noise(seed, width, height)
        with scope(device_type), self.model.ema_scope():
            for n in trange(iterations, desc='Generating'):
                x_T = None
                if self.variation_amount > 0:
                    seed_everything(seed)
                    target_noise = self.get_noise(width,height)
                    x_T = self.slerp(self.variation_amount, initial_noise, target_noise)
                elif initial_noise is not None:
                    # i.e. we specified particular variations
                    x_T = initial_noise
                else:
                    seed_everything(seed)
                    if self.model.device.type == 'mps':
                        x_T = self.get_noise(width,height)

                # make_image will do the equivalent of get_noise itself
                image = make_image(x_T)
                results.append([image, seed])
                if image_callback is not None:
                    image_callback(image, seed)
                seed = self.new_seed()
        return results
    
    def sample_to_image(self,samples):
        """
        Returns a function returning an image derived from the prompt and the initial image
        Return value depends on the seed at the time you call it
        """
        x_samples = self.model.decode_first_stage(samples)
        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
        if len(x_samples) != 1:
            raise Exception(
                f'>> expected to get a single image, but got {len(x_samples)}')
        x_sample = 255.0 * rearrange(
            x_samples[0].cpu().numpy(), 'c h w -> h w c'
        )
        return Image.fromarray(x_sample.astype(np.uint8))

    def generate_initial_noise(self, seed, width, height):
        initial_noise = None
        if self.variation_amount > 0 or len(self.with_variations) > 0:
            # use fixed initial noise plus random noise per iteration
            seed_everything(seed)
            initial_noise = self.get_noise(width,height)
            for v_seed, v_weight in self.with_variations:
                seed = v_seed
                seed_everything(seed)
                next_noise = self.get_noise(width,height)
                initial_noise = self.slerp(v_weight, initial_noise, next_noise)
            if self.variation_amount > 0:
                random.seed() # reset RNG to an actually random state, so we can get a random seed for variations
                seed = random.randrange(0,np.iinfo(np.uint32).max)
            return (seed, initial_noise)
        else:
            return (seed, None)

    # returns a tensor filled with random numbers from a normal distribution
    def get_noise(self,width,height):
        """
        Returns a tensor filled with random numbers, either form a normal distribution
        (txt2img) or from the latent image (img2img, inpaint)
        """
        raise NotImplementedError("get_noise() must be implemented in a descendent class")
    
    def new_seed(self):
        self.seed = random.randrange(0, np.iinfo(np.uint32).max)
        return self.seed

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
            v2 = torch.from_numpy(v2).to(self.model.device)

        return v2


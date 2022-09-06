'''
ldm.dream.generator.txt2img descends from ldm.dream.generator
'''

import torch
import numpy as  np
from ldm.dream.devices             import choose_autocast_device
from ldm.dream.generator.base      import Generator
from ldm.models.diffusion.ddim     import DDIMSampler

class Img2Img(Generator):
    def __init__(self,model):
        super().__init__(model)
        self.init_latent         = None    # by get_noise()
    
    @torch.no_grad()
    def get_make_image(self,prompt,sampler,steps,cfg_scale,ddim_eta,
                       conditioning,init_image,strength,step_callback=None,**kwargs):
        """
        Returns a function returning an image derived from the prompt and the initial image
        Return value depends on the seed at the time you call it.
        """

        # PLMS sampler not supported yet, so ignore previous sampler
        if not isinstance(sampler,DDIMSampler):
            print(
                f">> sampler '{sampler.__class__.__name__}' is not yet supported. Using DDIM sampler"
            )
            sampler = DDIMSampler(self.model, device=self.model.device)

        sampler.make_schedule(
            ddim_num_steps=steps, ddim_eta=ddim_eta, verbose=False
        )

        device_type,scope   = choose_autocast_device(self.model.device)
        with scope(device_type):
            self.init_latent = self.model.get_first_stage_encoding(
                self.model.encode_first_stage(init_image)
            ) # move to latent space

        t_enc = int(strength * steps)
        uc, c   = conditioning

        @torch.no_grad()
        def make_image(x_T):
            # encode (scaled latent)
            z_enc = sampler.stochastic_encode(
                self.init_latent,
                torch.tensor([t_enc]).to(self.model.device),
                noise=x_T
            )
            # decode it
            samples = sampler.decode(
                z_enc,
                c,
                t_enc,
                img_callback = step_callback,
                unconditional_guidance_scale=cfg_scale,
                unconditional_conditioning=uc,
            )
            return self.sample_to_image(samples)

        return make_image

    def get_noise(self,width,height):
        device      = self.model.device
        init_latent = self.init_latent
        assert init_latent is not None,'call to get_noise() when init_latent not set'
        if device.type == 'mps':
            return torch.randn_like(init_latent, device='cpu').to(device)
        else:
            return torch.randn_like(init_latent, device=device)

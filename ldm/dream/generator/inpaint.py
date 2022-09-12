'''
ldm.dream.generator.inpaint descends from ldm.dream.generator
'''

import torch
import numpy as  np
from einops import rearrange, repeat
from ldm.dream.devices             import choose_autocast_device
from ldm.dream.generator.img2img   import Img2Img
from ldm.models.diffusion.ddim     import DDIMSampler

class Inpaint(Img2Img):
    def __init__(self,model):
        self.init_latent = None
        super().__init__(model)
    
    @torch.no_grad()
    def get_make_image(self,prompt,sampler,steps,cfg_scale,ddim_eta,
                       conditioning,init_image,mask_image,strength,
                       step_callback=None,**kwargs):
        """
        Returns a function returning an image derived from the prompt and
        the initial image + mask.  Return value depends on the seed at
        the time you call it.  kwargs are 'init_latent' and 'strength'
        """

        mask_image = mask_image[0][0].unsqueeze(0).repeat(4,1,1).unsqueeze(0)
        mask_image = repeat(mask_image, '1 ... -> b ...', b=1)

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

        t_enc   = int(strength * steps)
        uc, c   = conditioning

        print(f">> target t_enc is {t_enc} steps")

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
                img_callback                 = step_callback,
                unconditional_guidance_scale = cfg_scale,
                unconditional_conditioning = uc,
                mask                       = mask_image,
                init_latent                = self.init_latent
            )
            return self.sample_to_image(samples)

        return make_image




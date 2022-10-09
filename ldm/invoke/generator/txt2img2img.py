'''
ldm.invoke.generator.txt2img inherits from ldm.invoke.generator
'''

import torch
import numpy as  np
import math
from ldm.invoke.generator.base  import Generator
from ldm.models.diffusion.ddim import DDIMSampler


class Txt2Img2Img(Generator):
    def __init__(self, model, precision):
        super().__init__(model, precision)
        self.init_latent = None    # for get_noise()

    @torch.no_grad()
    def get_make_image(self,prompt,sampler,steps,cfg_scale,ddim_eta,
                       conditioning,width,height,strength,step_callback=None,**kwargs):
        """
        Returns a function returning an image derived from the prompt and the initial image
        Return value depends on the seed at the time you call it
        kwargs are 'width' and 'height'
        """
        uc, c   = conditioning

        @torch.no_grad()
        def make_image(x_T):           
            
            trained_square = 512 * 512
            actual_square = width * height
            scale = math.sqrt(trained_square / actual_square)

            init_width = math.ceil(scale * width / 64) * 64
            init_height = math.ceil(scale * height / 64) * 64
            
            shape = [
                self.latent_channels,
                init_height // self.downsampling_factor,
                init_width // self.downsampling_factor,
            ]
            
            sampler.make_schedule(
                    ddim_num_steps=steps, ddim_eta=ddim_eta, verbose=False
            )
            
            #x = self.get_noise(init_width, init_height)
            x = x_T
            
            if self.free_gpu_mem and self.model.model.device != self.model.device:
                self.model.model.to(self.model.device)

            samples, _ = sampler.sample(
                batch_size                   = 1,
                S                            = steps,
                x_T                          = x,
                conditioning                 = c,
                shape                        = shape,
                verbose                      = False,
                unconditional_guidance_scale = cfg_scale,
                unconditional_conditioning   = uc,
                eta                          = ddim_eta,
                img_callback                 = step_callback
            )
            
            print(
                  f"\n>> Interpolating from {init_width}x{init_height} to {width}x{height} using DDIM sampling"
                 )
            
            # resizing
            samples = torch.nn.functional.interpolate(
                samples, 
                size=(height // self.downsampling_factor, width // self.downsampling_factor), 
                mode="bilinear"
            )

            t_enc = int(strength * steps)
            ddim_sampler = DDIMSampler(self.model, device=self.model.device)
            ddim_sampler.make_schedule(
                    ddim_num_steps=steps, ddim_eta=ddim_eta, verbose=False
            )

            z_enc = ddim_sampler.stochastic_encode(
                samples,
                torch.tensor([t_enc]).to(self.model.device),
                noise=self.get_noise(width,height,False)
            )

            # decode it
            samples = ddim_sampler.decode(
                z_enc,
                c,
                t_enc,
                img_callback = step_callback,
                unconditional_guidance_scale=cfg_scale,
                unconditional_conditioning=uc,
            )

            if self.free_gpu_mem:
                self.model.model.to("cpu")

            return self.sample_to_image(samples)

        return make_image


    # returns a tensor filled with random numbers from a normal distribution
    def get_noise(self,width,height,scale = True):
        # print(f"Get noise: {width}x{height}")
        if scale:
            trained_square = 512 * 512
            actual_square = width * height
            scale = math.sqrt(trained_square / actual_square)
            scaled_width = math.ceil(scale * width / 64) * 64
            scaled_height = math.ceil(scale * height / 64) * 64
        else:
            scaled_width = width
            scaled_height = height
            
        device      = self.model.device
        if device.type == 'mps':
            return torch.randn([1,
                                self.latent_channels,
                                scaled_height // self.downsampling_factor,
                                scaled_width  // self.downsampling_factor],
                                device='cpu').to(device)
        else:
            return torch.randn([1,
                                self.latent_channels,
                                scaled_height // self.downsampling_factor,
                                scaled_width  // self.downsampling_factor],
                                device=device)

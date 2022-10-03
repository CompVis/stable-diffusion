"""wrapper around part of Katherine Crowson's k-diffusion library, making it call compatible with other Samplers"""
import k_diffusion as K
import torch
import torch.nn as nn
from ldm.dream.devices import choose_torch_device
from ldm.models.diffusion.sampler import Sampler

class CFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, x, sigma, uncond, cond, cond_scale):
        x_in = torch.cat([x] * 2)
        sigma_in = torch.cat([sigma] * 2)
        cond_in = torch.cat([uncond, cond])
        uncond, cond = self.inner_model(x_in, sigma_in, cond=cond_in).chunk(2)
        return uncond + (cond - uncond) * cond_scale


class KSampler(Sampler):
    def __init__(self, model, schedule='lms', device=None, **kwargs):
        denoiser = K.external.CompVisDenoiser(model)
        super().__init__(
            denoiser,
            schedule,
            steps=model.num_timesteps,
        )
        self.ds    = None
        self.s_in  = None

        def forward(self, x, sigma, uncond, cond, cond_scale):
            x_in = torch.cat([x] * 2)
            sigma_in = torch.cat([sigma] * 2)
            cond_in = torch.cat([uncond, cond])
            uncond, cond = self.inner_model(
                x_in, sigma_in, cond=cond_in
            ).chunk(2)
            return uncond + (cond - uncond) * cond_scale

    def make_schedule(
            self,
            ddim_num_steps,
            ddim_discretize='uniform',
            ddim_eta=0.0,
            verbose=False,
    ):
        outer_model = self.model
        self.model  = outer_model.inner_model
        super().make_schedule(
            ddim_num_steps,
            ddim_discretize='uniform',
            ddim_eta=0.0,
            verbose=False,
        )
        self.model = outer_model
        self.ddim_num_steps = ddim_num_steps
        sigmas = K.sampling.get_sigmas_karras(
            n=ddim_num_steps,
            sigma_min=self.model.sigmas[0].item(),
            sigma_max=self.model.sigmas[-1].item(),
            rho=7.,
            device=self.device,
            # Birch-san recommends this, but it doesn't match the call signature in his branch of k-diffusion
            # concat_zero=False
        )
        self.sigmas = sigmas
        
    # ALERT: We are completely overriding the sample() method in the base class, which
    # means that inpainting will (probably?) not work correctly. To get this to work
    # we need to be able to modify the inner loop of k_heun, k_lms, etc, as is done
    # in an ugly way in the lstein/k-diffusion branch.
    
    # Most of these arguments are ignored and are only present for compatibility with
    # other samples
    @torch.no_grad()
    def sample(
        self,
        S,
        batch_size,
        shape,
        conditioning=None,
        callback=None,
        normals_sequence=None,
        img_callback=None,
        quantize_x0=False,
        eta=0.0,
        mask=None,
        x0=None,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
        verbose=True,
        x_T=None,
        log_every_t=100,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
        **kwargs,
    ):
        def route_callback(k_callback_values):
            if img_callback is not None:
                img_callback(k_callback_values['x'],k_callback_values['i'])

        # sigmas = self.model.get_sigmas(S)
        # sigmas are now set up in make_schedule - we take the last steps items
        sigmas = self.sigmas[-S:]
        if x_T is not None:
            x = x_T * sigmas[0]
        else:
            x = (
                torch.randn([batch_size, *shape], device=self.device)
                * sigmas[0]
            )   # for GPU draw
        model_wrap_cfg = CFGDenoiser(self.model)
        extra_args = {
            'cond': conditioning,
            'uncond': unconditional_conditioning,
            'cond_scale': unconditional_guidance_scale,
        }
        print(f'>> Sampling with k__{self.schedule}')
        return (
            K.sampling.__dict__[f'sample_{self.schedule}'](
                model_wrap_cfg, x, sigmas, extra_args=extra_args,
                callback=route_callback
            ),
            None,
        )

    @torch.no_grad()
    def p_sample(
            self,
            img,
            cond,
            ts,
            index,
            unconditional_guidance_scale=1.0,
            unconditional_conditioning=None,
            **kwargs,
    ):
        if self.model_wrap is None:
            self.model_wrap = CFGDenoiser(self.model)
        extra_args = {
            'cond': cond,
            'uncond': unconditional_conditioning,
            'cond_scale': unconditional_guidance_scale,
        }
        if self.s_in is None:
            self.s_in  = img.new_ones([img.shape[0]])
        if self.ds is None:
            self.ds = []

        # terrible, confusing names here
        steps = self.ddim_num_steps
        t_enc = self.t_enc
        
        # sigmas is a full steps in length, but t_enc might
        # be less. We start in the middle of the sigma array
        # and work our way to the end after t_enc steps.
        # index starts at t_enc and works its way to zero,
        # so the actual formula for indexing into sigmas:
        # sigma_index = (steps-index)
        s_index = t_enc - index - 1
        img =  K.sampling.__dict__[f'_{self.schedule}'](
            self.model_wrap,
            img,
            self.sigmas,
            s_index,
            s_in = self.s_in,
            ds   = self.ds,
            extra_args=extra_args,
        )

        return img, None, None

    def get_initial_image(self,x_T,shape,steps):
        if x_T is not None:
            return x_T + x_T * self.sigmas[0]
        else:
            return (torch.randn(shape, device=self.device) * self.sigmas[0])
        
    def prepare_to_sample(self,t_enc):
        self.t_enc      = t_enc
        self.model_wrap = None
        self.ds         = None
        self.s_in       = None

    def q_sample(self,x0,ts):
        '''
        Overrides parent method to return the q_sample of the inner model.
        '''
        return self.model.inner_model.q_sample(x0,ts)

    @torch.no_grad()
    def decode(
            self,
            z_enc,
            cond,
            t_enc,
            img_callback=None,
            unconditional_guidance_scale=1.0,
            unconditional_conditioning=None,
            use_original_steps=False,
            init_latent       = None,
            mask              = None,
    ):
        samples,_ = self.sample(
            batch_size = 1,
            S          = t_enc,
            x_T        = z_enc,
            shape      = z_enc.shape[1:],
            conditioning = cond,
            unconditional_guidance_scale=unconditional_guidance_scale,
            unconditional_conditioning = unconditional_conditioning,
            img_callback = img_callback,
            x0           = init_latent,
            mask         = mask
            )
        return samples

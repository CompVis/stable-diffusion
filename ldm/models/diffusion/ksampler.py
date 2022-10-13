"""wrapper around part of Katherine Crowson's k-diffusion library, making it call compatible with other Samplers"""
import k_diffusion as K
import torch
import torch.nn as nn
from ldm.invoke.devices import choose_torch_device
from ldm.models.diffusion.sampler import Sampler
from ldm.util import rand_perlin_2d
from ldm.modules.diffusionmodules.util import (
    make_ddim_sampling_parameters,
    make_ddim_timesteps,
    noise_like,
    extract_into_tensor,
)

def cfg_apply_threshold(result, threshold = 0.0, scale = 0.7):
    if threshold <= 0.0:
        return result
    maxval = 0.0 + torch.max(result).cpu().numpy()
    minval = 0.0 + torch.min(result).cpu().numpy()
    if maxval < threshold and minval > -threshold:
        return result
    if maxval > threshold:
        maxval = min(max(1, scale*maxval), threshold)
    if minval < -threshold:
        minval = max(min(-1, scale*minval), -threshold)
    return torch.clamp(result, min=minval, max=maxval)


class CFGDenoiser(nn.Module):
    def __init__(self, model, threshold = 0, warmup = 0):
        super().__init__()
        self.inner_model = model
        self.threshold = threshold
        self.warmup_max = warmup
        self.warmup = max(warmup / 10, 1)

    def forward(self, x, sigma, uncond, cond, cond_scale):
        x_in = torch.cat([x] * 2)
        sigma_in = torch.cat([sigma] * 2)
        cond_in = torch.cat([uncond, cond])
        uncond, cond = self.inner_model(x_in, sigma_in, cond=cond_in).chunk(2)
        if self.warmup < self.warmup_max:
            thresh = max(1, 1 + (self.threshold - 1) * (self.warmup / self.warmup_max))
            self.warmup += 1
        else:
            thresh = self.threshold
        if thresh > self.threshold:
            thresh = self.threshold
        return cfg_apply_threshold(uncond + (cond - uncond) * cond_scale, thresh)


class KSampler(Sampler):
    def __init__(self, model, schedule='lms', device=None, **kwargs):
        denoiser = K.external.CompVisDenoiser(model)
        super().__init__(
            denoiser,
            schedule,
            steps=model.num_timesteps,
        )
        self.sigmas = None
        self.ds     = None
        self.s_in   = None

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
        self.model          = outer_model
        self.ddim_num_steps = ddim_num_steps
        # we don't need both of these sigmas, but storing them here to make
        # comparison easier later on
        self.model_sigmas  = self.model.get_sigmas(ddim_num_steps)
        self.karras_sigmas = K.sampling.get_sigmas_karras(
            n=ddim_num_steps,
            sigma_min=self.model.sigmas[0].item(),
            sigma_max=self.model.sigmas[-1].item(),
            rho=7.,
            device=self.device,
        )
        self.sigmas = self.model_sigmas
        #self.sigmas = self.karras_sigmas
        
    # ALERT: We are completely overriding the sample() method in the base class, which
    # means that inpainting will not work. To get this to work we need to be able to
    # modify the inner loop of k_heun, k_lms, etc, as is done in an ugly way
    # in the lstein/k-diffusion branch.
    
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

    # this is a no-op, provided here for compatibility with ddim and plms samplers
    @torch.no_grad()
    def stochastic_encode(self, x0, t, use_original_steps=False, noise=None):
        return x0
    
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
        threshold = 0,
        perlin = 0,
        # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
        **kwargs,
    ):
        def route_callback(k_callback_values):
            if img_callback is not None:
                img_callback(k_callback_values['x'],k_callback_values['i'])

        # if make_schedule() hasn't been called, we do it now
        if self.sigmas is None:
            self.make_schedule(
                ddim_num_steps=S,
                ddim_eta = eta,
                verbose = False,
            )

        # sigmas are set up in make_schedule - we take the last steps items
        total_steps = len(self.sigmas)
        sigmas = self.sigmas[-S-1:]

        # x_T is variation noise. When an init image is provided (in x0) we need to add
        # more randomness to the starting image.
        if x_T is not None:
            if x0 is not None:
                x = x_T + torch.randn_like(x0, device=self.device) * sigmas[0]
            else:
                x = x_T * sigmas[0]
        else:
            x = torch.randn([batch_size, *shape], device=self.device) * sigmas[0]

        model_wrap_cfg = CFGDenoiser(self.model, threshold=threshold, warmup=max(0.8*S,S-10))
        extra_args = {
            'cond': conditioning,
            'uncond': unconditional_conditioning,
            'cond_scale': unconditional_guidance_scale,
        }
        print(f'>> Sampling with k_{self.schedule} starting at step {len(self.sigmas)-S-1} of {len(self.sigmas)-1} ({S} new sampling steps)')
        return (
            K.sampling.__dict__[f'sample_{self.schedule}'](
                model_wrap_cfg, x, sigmas, extra_args=extra_args,
                callback=route_callback
            ),
            None,
        )

    # this code will support inpainting if and when ksampler API modified or
    # a workaround is found.
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

    # REVIEW THIS METHOD: it has never been tested. In particular,
    # we should not be multiplying by self.sigmas[0] if we
    # are at an intermediate step in img2img. See similar in
    # sample() which does work.
    def get_initial_image(self,x_T,shape,steps):
        print(f'WARNING: ksampler.get_initial_image(): get_initial_image needs testing')
        x = (torch.randn(shape, device=self.device) * self.sigmas[0])
        if x_T is not None:
            return x_T + x
        else:
            return x
        
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


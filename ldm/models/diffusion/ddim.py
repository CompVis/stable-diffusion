"""SAMPLING ONLY."""

import torch
import numpy as np
from tqdm import tqdm
from functools import partial
from ldm.dream.devices import choose_torch_device
from ldm.models.diffusion.sampler import Sampler
from ldm.modules.diffusionmodules.util import  noise_like

class DDIMSampler(Sampler):
    def __init__(self, model, schedule='linear', device=None, **kwargs):
        super().__init__(model,schedule,model.num_timesteps,device)

    # This is the central routine
    @torch.no_grad()
    def p_sample(
            self,
            x,
            c,
            t,
            index,
            repeat_noise=False,
            use_original_steps=False,
            quantize_denoised=False,
            temperature=1.0,
            noise_dropout=0.0,
            score_corrector=None,
            corrector_kwargs=None,
            unconditional_guidance_scale=1.0,
            unconditional_conditioning=None,
            **kwargs,
    ):
        b, *_, device = *x.shape, x.device

        if (
            unconditional_conditioning is None
            or unconditional_guidance_scale == 1.0
        ):
            e_t = self.model.apply_model(x, t, c)
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            c_in = torch.cat([unconditional_conditioning, c])
            e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
            e_t = e_t_uncond + unconditional_guidance_scale * (
                e_t - e_t_uncond
            )

        if score_corrector is not None:
            assert self.model.parameterization == 'eps'
            e_t = score_corrector.modify_score(
                self.model, e_t, x, t, c, **corrector_kwargs
            )

        alphas = (
            self.model.alphas_cumprod
            if use_original_steps
            else self.ddim_alphas
        )
        alphas_prev = (
            self.model.alphas_cumprod_prev
            if use_original_steps
            else self.ddim_alphas_prev
        )
        sqrt_one_minus_alphas = (
            self.model.sqrt_one_minus_alphas_cumprod
            if use_original_steps
            else self.ddim_sqrt_one_minus_alphas
        )
        sigmas = (
            self.model.ddim_sigmas_for_original_num_steps
            if use_original_steps
            else self.ddim_sigmas
        )
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full(
            (b, 1, 1, 1), sqrt_one_minus_alphas[index], device=device
        )

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        # direction pointing to x_t
        dir_xt = (1.0 - a_prev - sigma_t**2).sqrt() * e_t
        noise = (
            sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        )
        if noise_dropout > 0.0:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0, None


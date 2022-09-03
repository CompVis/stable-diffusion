"""wrapper around part of Katherine Crowson's k-diffusion library, making it call compatible with other Samplers"""
import k_diffusion as K
import torch
import torch.nn as nn
from ldm.dream.devices import choose_torch_device
from ldm.modules.diffusionmodules.util import rand_perlin_2d

def cfg_apply_threshold(result, threshold = 0.0, scale = 0.707):
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
    def __init__(self, model, threshold = 0):
        super().__init__()
        self.inner_model = model
        self.threshold = threshold

    def forward(self, x, sigma, uncond, cond, cond_scale):
        x_in = torch.cat([x] * 2)
        sigma_in = torch.cat([sigma] * 2)
        cond_in = torch.cat([uncond, cond])
        uncond, cond = self.inner_model(x_in, sigma_in, cond=cond_in).chunk(2)
        return cfg_apply_threshold(uncond + (cond - uncond) * cond_scale, self.threshold)


class KSampler(object):
    def __init__(self, model, schedule='lms', device=None, **kwargs):
        super().__init__()
        self.model = K.external.CompVisDenoiser(model)
        self.schedule = schedule
        self.device   = device or choose_torch_device()
        #self.threshold = threshold or 0

        def forward(self, x, sigma, uncond, cond, cond_scale):
            x_in = torch.cat([x] * 2)
            sigma_in = torch.cat([sigma] * 2)
            cond_in = torch.cat([uncond, cond])
            uncond, cond = self.inner_model(
                x_in, sigma_in, cond=cond_in
            ).chunk(2)
            return uncond + (cond - uncond) * cond_scale
            #return cfg_apply_threshold(uncond + (cond - uncond) * cond_scale, self.threshold)



    # most of these arguments are ignored and are only present for compatibility with
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
                img_callback(k_callback_values['x'], k_callback_values['i'])

        sigmas = self.model.get_sigmas(S)
        if x_T is not None:
            x = x_T * sigmas[0]
        else:
            x = (
                torch.randn([batch_size, *shape], device=self.device)
                * sigmas[0]
            )   # for GPU draw

        if perlin > 0.0:
            print(shape)
            x = (1 - perlin / 2) * x + perlin * rand_perlin_2d((shape[1], shape[2]), (8, 8)).to(self.device)

        model_wrap_cfg = CFGDenoiser(self.model, threshold=threshold)
        extra_args = {
            'cond': conditioning,
            'uncond': unconditional_conditioning,
            'cond_scale': unconditional_guidance_scale,
        }
        return (
            K.sampling.__dict__[f'sample_{self.schedule}'](
                model_wrap_cfg, x, sigmas, extra_args=extra_args,
                callback=route_callback
            ),
            None,
        )

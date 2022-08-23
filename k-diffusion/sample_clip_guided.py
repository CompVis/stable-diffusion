#!/usr/bin/env python3

"""CLIP guided sampling from k-diffusion models."""

import argparse
import math

import accelerate
import clip
from kornia import augmentation as KA
from resize_right import resize
import torch
from torch.nn import functional as F
from torchvision import transforms
from tqdm import trange, tqdm

import k_diffusion as K


def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)


def make_cond_model_fn(model, cond_fn):
    def model_fn(x, sigma, **kwargs):
        with torch.enable_grad():
            x = x.detach().requires_grad_()
            denoised = model(x, sigma, **kwargs)
            cond_grad = cond_fn(x, sigma, denoised=denoised, **kwargs).detach()
            cond_denoised = denoised.detach() + cond_grad * K.utils.append_dims(sigma**2, x.ndim)
        return cond_denoised
    return model_fn


def make_static_thresh_model_fn(model, value=1.):
    def model_fn(x, sigma, **kwargs):
        return model(x, sigma, **kwargs).clamp(-value, value)
    return model_fn


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('prompt', type=str,
                   default='the prompt to use')
    p.add_argument('--batch-size', type=int, default=16,
                   help='the batch size')
    p.add_argument('--checkpoint', type=str, required=True,
                   help='the checkpoint to use')
    p.add_argument('--churn', type=float, default=50.,
                   help='the amount of noise to add during sampling')
    p.add_argument('--clip-guidance-scale', '-cgs', type=float, default=500.,
                   help='the CLIP guidance scale')
    p.add_argument('--clip-model', type=str, default='ViT-B/16', choices=clip.available_models(),
                   help='the CLIP model to use')
    p.add_argument('--config', type=str, required=True,
                   help='the model config')
    p.add_argument('-n', type=int, default=64,
                   help='the number of images to sample')
    p.add_argument('--prefix', type=str, default='out',
                   help='the output prefix')
    p.add_argument('--steps', type=int, default=100,
                   help='the number of denoising steps')
    args = p.parse_args()

    config = K.config.load_config(open(args.config))
    model_config = config['model']
    # TODO: allow non-square input sizes
    assert len(model_config['input_size']) == 2 and model_config['input_size'][0] == model_config['input_size'][1]
    size = model_config['input_size']

    accelerator = accelerate.Accelerator()
    device = accelerator.device
    print('Using device:', device, flush=True)

    inner_model = K.config.make_model(config).eval().requires_grad_(False).to(device)
    inner_model.load_state_dict(torch.load(args.checkpoint, map_location='cpu')['model_ema'])
    accelerator.print('Parameters:', K.utils.n_params(inner_model))
    model = K.Denoiser(inner_model, sigma_data=model_config['sigma_data'])

    sigma_min = model_config['sigma_min']
    sigma_max = model_config['sigma_max']

    clip_model = clip.load(args.clip_model, device=device)[0].eval().requires_grad_(False)
    clip_normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                          std=(0.26862954, 0.26130258, 0.27577711))
    clip_size = (clip_model.visual.input_resolution, clip_model.visual.input_resolution)
    aug = KA.RandomAffine(0, (1/14, 1/14), p=1, padding_mode='border')

    def get_image_embed(x):
        if x.shape[2:4] != clip_size:
            x = resize(x, out_shape=clip_size, pad_mode='reflect')
        x = clip_normalize(x)
        x = clip_model.encode_image(x).float()
        return F.normalize(x)

    target_embed = F.normalize(clip_model.encode_text(clip.tokenize(args.prompt, truncate=True).to(device)).float())

    def cond_fn(x, t, denoised):
        image_embed = get_image_embed(aug(denoised.add(1).div(2)))
        loss = spherical_dist_loss(image_embed, target_embed).sum() * args.clip_guidance_scale
        grad = -torch.autograd.grad(loss, x)[0]
        return grad

    model_fn = make_cond_model_fn(model, cond_fn)
    model_fn = make_static_thresh_model_fn(model_fn)

    @torch.no_grad()
    @K.utils.eval_mode(model)
    def run():
        if accelerator.is_local_main_process:
            tqdm.write('Sampling...')
        sigmas = K.sampling.get_sigmas_karras(args.steps, sigma_min, sigma_max, rho=7., device=device)
        def sample_fn(n):
            x = torch.randn([n, model_config['input_channels'], size[0], size[1]], device=device) * sigmas[0]
            x_0 = K.sampling.sample_dpm_2(model_fn, x, sigmas, s_churn=args.churn, disable=not accelerator.is_local_main_process)
            return x_0
        x_0 = K.evaluation.compute_features(accelerator, sample_fn, lambda x: x, args.n, args.batch_size)
        if accelerator.is_main_process:
            for i, out in enumerate(x_0):
                filename = f'{args.prefix}_{i:05}.png'
                K.utils.to_pil_image(out).save(filename)

    try:
        run()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()

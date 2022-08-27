"""make variations of input image"""

import argparse, os, sys, glob
import PIL
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange, repeat
from torchvision.utils import make_grid
from torch import autocast
from contextlib import nullcontext
import time
from pytorch_lightning import seed_everything
import imutil

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    #model = model.half()
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def load_img(path):
    image = Image.open(path).convert("RGB")
    w, h = image.size
    print(f"loaded input image of size ({w}, {h}) from {path}")
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).half()
    return 2.*image - 1.


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt1",
        type=str,
        nargs="?",
        default="8 k resolution, alphonse mucha, artbook, artgerm, bo chen, jin xiaodi , art by WLOP, Artgerm, Alphonse Mucha, artstation, concept art, detailed and intricate environment, digital painting, dramatic lighting, elegant, ferdinand knab, global illumination, highly detailed, illustration, ilya kuvshinov, intricate, loish, makoto shinkai, lois van baarle, masterpiece, matte, octane render, promo art, radiant light, rhads, rossdraws, sharp focus, smooth, splash art, tom bagshaw, unreal engine 5, vivid vibrant, wallpaper, wide angle",
        help="words describing the duck"
    )
    parser.add_argument(
        "--prompt2",
        type=str,
        nargs="?",
        #default="Greg Rutkowski painting a self-portrait of Grzegorz Rutkowski in the style of " * 3,
        default=None,
        help="words describing the rabbit"
    )

    parser.add_argument(
        "--init-img",
        type=str,
        nargs="?",
        help="path to the rabbit image"
    )
    parser.add_argument(
        "--init-img2",
        type=str,
        nargs="?",
        help="path to the duck image"
    )

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="output"
    )

    parser.add_argument(
        '--slerpradius',
        type=float,
        nargs="?",
        help="0 < slerpradius <= 1",
        default=1.0,
    )

    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )

    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save indiviual samples. For speed measurements.",
    )

    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )

    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across all samples ",
    )

    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor, most often 8 or 16",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given prompt. A.k.a batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=5.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )

    parser.add_argument(
        "--strength",
        type=float,
        default=0.75,
        help="strength for noising/unnoising. 1.0 corresponds to full destruction of information in init image",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed1",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--seed2",
        type=int,
        default=43,
        help="the seed of the destination",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    parser.add_argument(
        "--output_video",
        type=str,
        help="output video filename",
    )

    opt = parser.parse_args()
    seed_everything(opt.seed1)

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}").half()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if opt.plms:
        raise NotImplementedError("PLMS sampler not (yet) supported")
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    batch_size = opt.n_samples
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1

    assert os.path.isfile(opt.init_img)
    init_image1 = load_img(opt.init_img).to(device)
    init_image1 = repeat(init_image1, '1 ... -> b ...', b=batch_size)

    if init_image2:
        init_image2 = load_img(opt.init_img2).to(device)
        init_image2 = repeat(init_image2, '1 ... -> b ...', b=batch_size)
    else:
        init_image2 = init_image1

    init_latent1 = model.get_first_stage_encoding(model.encode_first_stage(init_image1))
    init_latent2 = model.get_first_stage_encoding(model.encode_first_stage(init_image2))


    # move latents to the left to the left
    #init_latent = torch.roll(init_latent, shifts=-1, dims=-1)

    sampler.make_schedule(ddim_num_steps=opt.ddim_steps, ddim_eta=opt.ddim_eta, verbose=False)

    if not opt.prompt2:
        opt.prompt2 = opt.prompt1
    else:
        assert False

    assert 0. <= opt.strength <= 1., 'can only work with strength in [0.0, 1.0]'
    t_enc = int(opt.strength * opt.ddim_steps)
    print(f"target t_enc is {t_enc} steps")

    if opt.output_video:
        vid = imutil.Video(filename=opt.output_video)
    b, chan, height, width = init_latent1.shape
    np.random.seed(opt.seed1)
    noise1 = np.random.normal(0, 1, (1, chan, height, width))
    np.random.seed(opt.seed2)
    noise2 = np.random.normal(0, 1, (1, chan, height, width))

    precision_scope = autocast if opt.precision == "autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                tic = time.time()
                all_samples = list()
                for n in trange(opt.n_iter, desc="Sampling"):
                    theta = opt.slerpradius * (.5 - .5 * np.cos(np.pi * n / opt.n_iter))
                    print(f'Theta is {theta}')

                    # https://www.inference.vc/high-dimensional-gaussian-distributions-are-soap-bubble/
                    #noise = torch.tensor((theta**.5) * noise1 + ((1 - theta)**.5) * noise2).to(device).half()
                    noise = torch.tensor(noise1).to(device).half()

                    uc = None
                    if opt.scale != 1.0:
                        uc = model.get_learned_conditioning(batch_size * [""])

                    # We have two vectors, c1 and c2, representing thought vectors
                    c1 = model.get_learned_conditioning(opt.prompt1)
                    c2 = model.get_learned_conditioning(opt.prompt2)

                    # Linear interpolation?
                    #c = theta * c2 + (1 - theta) * c1

                    # Linear interpolation in polar coordinates?
                    #c = theta**.5 * c2 + (1 - theta)**.5 * c1

                    # A wipe along the long axis?
                    i = int(theta * c1.shape[-1] + 0.499)
                    c = torch.concat([c2[::, :, :i], c1[::, :, i:]], dim=-1)

                    # Wipe along the thin axis?
                    #i = int(theta * c1.shape[1] + 0.499)
                    #c = torch.concat([c1[:, :i], c2[:, i:]], dim=1)

                    init_latent = theta * init_latent2 + (1 - theta) * init_latent1

                    # encode (scaled latent)
                    z_enc = sampler.stochastic_encode(init_latent,
                            torch.tensor([t_enc]*batch_size).to(device),
                            noise=noise)
                    # decode it
                    samples = sampler.decode(z_enc, c, t_enc, unconditional_guidance_scale=opt.scale,
                                                unconditional_conditioning=uc, init_latent=init_latent)

                    x_samples = model.decode_first_stage(samples)
                    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                    if opt.output_video:
                        vid.add_frame(x_samples)

                    if not opt.skip_save:
                        for x_sample in x_samples:
                            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                            Image.fromarray(x_sample.astype(np.uint8)).save(
                                os.path.join(sample_path, f"{base_count:05}.png"))
                            base_count += 1
                    all_samples.append(x_samples)

                if not opt.skip_grid:
                    # additionally, save as grid
                    grid = torch.stack(all_samples, 0)
                    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                    grid = make_grid(grid, nrow=n_rows)

                    # to image
                    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                    Image.fromarray(grid.astype(np.uint8)).save(os.path.join(outpath, f'grid-{grid_count:04}.png'))
                    grid_count += 1

                toc = time.time()

    print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")
    if opt.output_video:
        vid.finish()


if __name__ == "__main__":
    main()

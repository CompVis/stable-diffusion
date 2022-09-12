import argparse, os, re
import torch
import numpy as np
from random import randint
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext
from einops import rearrange, repeat
import accelerate
from ldm.util import instantiate_from_config
from optimUtils import split_weighted_subprompts, logger
from transformers import logging
import pandas as pd

from prompt_parser import PromptParser

logging.set_verbosity_error()

def patch_conv(**patch):
    cls = torch.nn.Conv2d
    init = cls.__init__
    def __init__(self, *args, **kwargs):
        return init(self, *args, **kwargs, **patch)
    cls.__init__ = __init__

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    return sd


def load_img(path):
    image = Image.open(path).convert("RGB")
    w, h = image.size
    print(f"loaded input image of size ({w}, {h}) from {path}")
    w, h = map(lambda x: x - x % 64, (w, h))
    image = image.resize((w, h), resample=Image.Resampling.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.


config = "scripts/v1-inference.yaml"
ckpt = "model.ckpt"

parser = argparse.ArgumentParser()

parser.add_argument(
    "--prompt", type=str, nargs="?", default="a painting of a virus monster playing guitar", help="the prompt to render"
)
parser.add_argument("--outdir", type=str, nargs="?", help="dir to write results to", default="temp")
parser.add_argument("--init_img", type=str, nargs="?", help="path to the input image", default="temp/input.png")

parser.add_argument(
    "--tiling",
    type=str,
    default="false",
    help="Tiles the generated image",
)

parser.add_argument(
    "--ddim_steps",
    type=int,
    default=50,
    help="number of ddim sampling steps",
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
    "--H",
    type=int,
    default=None,
    help="image height, in pixel space",
)
parser.add_argument(
    "--W",
    type=int,
    default=None,
    help="image width, in pixel space",
)
parser.add_argument(
    "--strength",
    type=float,
    default=0.75,
    help="strength for noising/unnoising. 1.0 corresponds to full destruction of information in init image",
)
parser.add_argument(
    "--n_samples",
    type=int,
    default=1,
    help="how many samples to produce for each given prompt. A.k.a. batch size",
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
    default=7.5,
    help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
)
parser.add_argument(
    "--from-file",
    type=str,
    help="if specified, load prompts from this file",
)
parser.add_argument(
    "--seed",
    type=int,
    default=None,
    help="the seed (for reproducible sampling)",
)
parser.add_argument(
    "--device",
    type=str,
    default="cuda",
    help="CPU or GPU (cuda/cuda:0/cuda:1/...)",
)
parser.add_argument(
    "--unet_bs",
    type=int,
    default=1,
    help="Slightly reduces inference time at the expense of high VRAM (value > 1 not recommended )",
)
parser.add_argument(
    "--turbo",
    action="store_true",
    help="Reduces inference time on the expense of 1GB VRAM",
)
parser.add_argument(
    "--precision", type=str, help="evaluate at this precision", choices=["full", "autocast"], default="autocast"
)

parser.add_argument(
    "--sampler",
    type=str,
    help="sampler",
    choices=["ddim"],
    default="ddim",
)
opt = parser.parse_args()

tic = time.time()
os.makedirs(opt.outdir, exist_ok=True)
outpath = opt.outdir

accelerator = accelerate.Accelerator()
device = accelerator.device
seed_everything(opt.seed)
seeds = torch.randint(-2 ** 63, 2 ** 63 - 1, [accelerator.num_processes])
torch.manual_seed(seeds[accelerator.process_index].item())

if opt.tiling == "true":
    patch_conv(padding_mode='circular')
    print("patched for tiling")

sd = load_model_from_config(f"{ckpt}")
li, lo = [], []
for key, value in sd.items():
    sp = key.split(".")
    if (sp[0]) == "model":
        if "input_blocks" in sp:
            li.append(key)
        elif "middle_block" in sp:
            li.append(key)
        elif "time_embed" in sp:
            li.append(key)
        else:
            lo.append(key)
for key in li:
    sd["model1." + key[6:]] = sd.pop(key)
for key in lo:
    sd["model2." + key[6:]] = sd.pop(key)

config = OmegaConf.load(f"{config}")

assert os.path.isfile(opt.init_img)
init_image = load_img(opt.init_img).to(opt.device)

model = instantiate_from_config(config.modelUNet)
_, _ = model.load_state_dict(sd, strict=False)
model.eval()
model.cdevice = opt.device
model.unet_bs = opt.unet_bs
model.turbo = opt.turbo

modelCS = instantiate_from_config(config.modelCondStage)
_, _ = modelCS.load_state_dict(sd, strict=False)
modelCS.eval()
modelCS.cond_stage_model.device = opt.device

modelFS = instantiate_from_config(config.modelFirstStage)
_, _ = modelFS.load_state_dict(sd, strict=False)
modelFS.eval()
del sd
if opt.device != "cpu" and opt.precision == "autocast":
    model.half()
    modelCS.half()
    modelFS.half()
    init_image = init_image.half()

batch_size = opt.n_samples
if not opt.from_file:
    assert opt.prompt is not None
    prompt = opt.prompt
    data = [batch_size * [prompt]]

else:
    print(f"reading prompts from {opt.from_file}")
    with open(opt.from_file, "r") as f:
        data = f.read().splitlines()
        data = batch_size * list(data)
        data = list(chunk(sorted(data), batch_size))

modelFS.to(opt.device)

init_image = repeat(init_image, "1 ... -> b ...", b=batch_size)
init_latent = modelFS.get_first_stage_encoding(modelFS.encode_first_stage(init_image))  # move to latent space

if opt.device != "cpu":
    mem = torch.cuda.memory_allocated() / 1e6
    modelFS.to("cpu")
    while torch.cuda.memory_allocated() / 1e6 >= mem:
        time.sleep(1)


assert 0.0 <= opt.strength <= 1.0, "can only work with strength in [0.0, 1.0]"
t_enc = int(opt.strength * opt.ddim_steps)
print(f"target t_enc is {t_enc} steps")


if opt.precision == "autocast" and opt.device != "cpu":
    precision_scope = autocast
else:
    precision_scope = nullcontext

seeds = ""
prompt_parser = PromptParser(modelCS)
with torch.no_grad():
    for n in trange(opt.n_iter, desc="Sampling"):
        for prompts in tqdm(data, desc="data"):

            with precision_scope("cuda"):
                modelCS.to(opt.device)
                uc = None
                if opt.scale != 1.0:
                    uc = modelCS.get_learned_conditioning(batch_size * [""])
                if isinstance(prompts, tuple):
                    prompts = list(prompts)

                prompt_guidance = prompt_parser.get_prompt_guidance(prompts[0], opt.ddim_steps, batch_size)

                c = prompt_guidance[0]

                if opt.device != "cpu":
                    mem = torch.cuda.memory_allocated() / 1e6
                    modelCS.to("cpu")
                    while torch.cuda.memory_allocated() / 1e6 >= mem:
                        time.sleep(1)

                # encode (scaled latent)
                z_enc = model.stochastic_encode(
                    init_latent,
                    torch.tensor([t_enc] * batch_size).to(opt.device),
                    opt.seed,
                    opt.ddim_eta,
                    opt.ddim_steps,
                )
                # decode it
                samples_ddim = model.sample(
                    t_enc,
                    c,
                    z_enc,
                    unconditional_guidance_scale=opt.scale,
                    unconditional_conditioning=uc,
                    sampler = opt.sampler,
                    prompt_guidance=prompt_guidance
                )

                modelFS.to(opt.device)
                print("saving images")
                for i in range(batch_size):

                    x_samples_ddim = modelFS.decode_first_stage(samples_ddim[i].unsqueeze(0))
                    x_sample = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                    x_sample = 255.0 * rearrange(x_sample[0].cpu().numpy(), "c h w -> h w c")
                    Image.fromarray(x_sample.astype(np.uint8)).save(
                        os.path.join(outpath, "temp.png")
                    )
                    seeds += str(opt.seed) + ","
                    opt.seed += 1

                if opt.device != "cpu":
                    mem = torch.cuda.memory_allocated() / 1e6
                    modelFS.to("cpu")
                    while torch.cuda.memory_allocated() / 1e6 >= mem:
                        time.sleep(1)

                del samples_ddim

toc = time.time()
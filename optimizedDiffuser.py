
import argparse, os, sys, glob, random
import torch
import numpy as np
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
from ldm.util import instantiate_from_config
from ldm.models.diffusion.plms import PLMSSampler


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


config = "optimizedSD/v1-inference.yaml"
configFS = "optimizedSD/firstStage.yml"
configCS = "optimizedSD/condStage.yml"
ckpt = "models/ldm/stable-diffusion-v1/model.ckpt"
device = "cuda"

parser = argparse.ArgumentParser()

parser.add_argument(
    "--prompt",
    type=str,
    nargs="?",
    default="a painting of a virus monster playing guitar",
    help="the prompt to render"
)
parser.add_argument(
    "--outdir",
    type=str,
    nargs="?",
    help="dir to write results to",
    default="outputs/txt2img-samples"
)
parser.add_argument(
    "--skip_grid",
    action='store_true',
    help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
)
parser.add_argument(
    "--skip_save",
    action='store_true',
    help="do not save individual samples. For speed measurements.",
)
parser.add_argument(
    "--ddim_steps",
    type=int,
    default=50,
    help="number of ddim sampling steps",
)

parser.add_argument(
    "--fixed_code",
    action='store_true',
    help="if enabled, uses the same starting code across samples ",
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
    default=512,
    help="image height, in pixel space",
)
parser.add_argument(
    "--W",
    type=int,
    default=512,
    help="image width, in pixel space",
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
    help="downsampling factor",
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
    default=42,
    help="the seed (for reproducible sampling)",
)
opt = parser.parse_args()

os.makedirs(opt.outdir, exist_ok=True)
outpath = opt.outdir

sample_path = os.path.join(outpath, "samples")
os.makedirs(sample_path, exist_ok=True)
base_count = len(os.listdir(sample_path))
grid_count = len(os.listdir(outpath)) - 1
seed_everything(opt.seed)
config = OmegaConf.load(f"{config}")
sd = load_model_from_config(f"{ckpt}")

model = instantiate_from_config(config.model)
m, u = model.load_state_dict(sd, strict=False)
model.eval()
model = model.to(device)
sampler = PLMSSampler(model)

li = []
for key, value in sd.items():
    if(key.split('.')[0]) == 'model':
        li.append(key)
for key in li:
    sd[key[6:]] = sd.pop(key)
model.sd = sd


configCS = OmegaConf.load(f"{configCS}")
modelCS = instantiate_from_config(configCS.model)
_, _ = modelCS.load_state_dict(sd, strict=False)
modelCS.eval()
modelCS = modelCS.to(device)

precision = "full"

start_code = None
if opt.fixed_code:
    start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)


batch_size = opt.n_samples
n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
if not opt.from_file:
    prompt = opt.prompt
    assert prompt is not None
    data = [batch_size * [prompt]]

else:
    print(f"reading prompts from {opt.from_file}")
    with open(opt.from_file, "r") as f:
        data = f.read().splitlines()
        data = list(chunk(data, batch_size))


precision_scope = autocast if (device == "cuda" and precision=="autocast") else nullcontext
with torch.no_grad():
    with precision_scope("cuda"):
        tic = time.time()
        all_samples = list()
        for n in trange(opt.n_iter, desc="Sampling"):
            for prompts in tqdm(data, desc="data"):
                uc = None
                if opt.scale != 1.0:
                    uc = modelCS.get_learned_conditioning(batch_size * [""])
                if isinstance(prompts, tuple):
                    prompts = list(prompts)


                c = modelCS.get_learned_conditioning(prompts)
                print(c.shape)
                shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                mem = torch.cuda.memory_allocated()/1e6
                del modelCS
                while(torch.cuda.memory_allocated()/1e6 >= mem):
                    time.sleep(1)
                # print(torch.cuda.memory_allocated()/1e6)


                samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                conditioning=c,
                                batch_size=opt.n_samples,
                                shape=shape,
                                verbose=False,
                                unconditional_guidance_scale=opt.scale,
                                unconditional_conditioning=uc,
                                eta=opt.ddim_eta,
                                x_T=start_code)

                mem = torch.cuda.memory_allocated()/1e6
                del model
                del sampler
                while(torch.cuda.memory_allocated()/1e6 >= mem):
                    time.sleep(1)
                # print(torch.cuda.memory_allocated()/1e6)

                
                configFS = OmegaConf.load(f"{configFS}")
                modelFS = instantiate_from_config(configFS.model)
                _, _ = modelFS.load_state_dict(sd, strict=False)
                modelFS.eval()
                modelFS = modelFS.to(device)

                x_samples_ddim = modelFS.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                
                mem = torch.cuda.memory_allocated()/1e6
                del modelFS
                while(torch.cuda.memory_allocated()/1e6 >= mem):
                    time.sleep(1)

                # if not skip_save:
                print("saving image")
                for x_sample in x_samples_ddim:
                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                    Image.fromarray(x_sample.astype(np.uint8)).save(
                        os.path.join(sample_path, "_".join(opt.prompt.split()) + "_" + f"{base_count:05}.png"))
                    base_count += 1

                if not opt.skip_grid:
                    all_samples.append(x_samples_ddim)
                
                del x_samples_ddim
                print(torch.cuda.memory_allocated()/1e6)

        # if not skip_grid:
        #     # additionally, save as grid
        #     grid = torch.stack(all_samples, 0)
        #     grid = rearrange(grid, 'n b c h w -> (n b) c h w')
        #     grid = make_grid(grid, nrow=n_rows)

        #     # to image
        #     grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
        #     Image.fromarray(grid.astype(np.uint8)).save(os.path.join(outpath, f'grid-{grid_count:04}.png'))
        #     grid_count += 1

        toc = time.time()


print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
        f" \nEnjoy.")

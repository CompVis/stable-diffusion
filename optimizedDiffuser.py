
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


skip_grid = "False"
skip_save = "False"
plms = True
fixed_code = "False"
laion400m = "False"
ddim_eta = 0.0
C = 4 #Latent channels
f = 8
n_samples = 1
n_rows = 0
scale = 7.5
outdir = "outputs/txt2img-samples"
config = "optimizedSD/v1-inference.yaml"
configFS = "optimizedSD/firstStage.yml"
configCS = "optimizedSD/condStage.yml"
ckpt = "models/ldm/stable-diffusion-v1/model.ckpt"
from_file = False

precision = "full"
# precision = "autocast"
device = "cuda"
seed = random.randint(0,100000)
n_iter = 1
ddim_steps = 50

os.makedirs(outdir, exist_ok=True)
outpath = outdir

sample_path = os.path.join(outpath, "samples")
os.makedirs(sample_path, exist_ok=True)
base_count = len(os.listdir(sample_path))
grid_count = len(os.listdir(outpath)) - 1
seed_everything(seed)
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

prompt = "Pikachu in New York City"
precision = "full"
W = 512
H = 256
if fixed_code:
    start_code = torch.randn([n_samples, C, H // f, W // f], device=device)


batch_size = n_samples
n_rows = n_rows if n_rows > 0 else batch_size
if not from_file:
    prompt = prompt
    assert prompt is not None
    data = [batch_size * [prompt]]

else:
    print(f"reading prompts from {from_file}")
    with open(from_file, "r") as f:
        data = f.read().splitlines()
        data = list(chunk(data, batch_size))


precision_scope = autocast if (device == "cuda" and precision=="autocast") else nullcontext
with torch.no_grad():
    with precision_scope("cuda"):
        # with model.ema_scope():
            tic = time.time()
            all_samples = list()
            for n in trange(n_iter, desc="Sampling"):
                for prompts in tqdm(data, desc="data"):
                    uc = None
                    if scale != 1.0:
                        uc = modelCS.get_learned_conditioning(batch_size * [""])
                    if isinstance(prompts, tuple):
                        prompts = list(prompts)
                    c = modelCS.get_learned_conditioning(prompts)
                    print(c.shape)
                    shape = [C, H // f, W // f]
                    mem = torch.cuda.memory_allocated()/1e6
                    del modelCS
                    while(torch.cuda.memory_allocated()/1e6 >= mem):
                        time.sleep(1)
                    print(torch.cuda.memory_allocated()/1e6)

                    samples_ddim, _ = sampler.sample(S=ddim_steps,
                                    conditioning=c,
                                    batch_size=n_samples,
                                    shape=shape,
                                    verbose=False,
                                    unconditional_guidance_scale=scale,
                                    unconditional_conditioning=uc,
                                    eta=ddim_eta,
                                    x_T=start_code)

                    mem = torch.cuda.memory_allocated()/1e6
                    del model
                    del sampler
                    while(torch.cuda.memory_allocated()/1e6 >= mem):
                        time.sleep(1)
                    print(torch.cuda.memory_allocated()/1e6)

                    
                    configFS = OmegaConf.load(f"{configFS}")
                    modelFS = instantiate_from_config(configFS.model)
                    _, _ = modelFS.load_state_dict(sd, strict=False)
                    modelFS.eval()
                    modelFS = modelFS.to(device)


                    x_samples_ddim = modelFS.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                    
                    print(torch.cuda.memory_allocated()/1e6)
                    del modelFS
                    print(torch.cuda.memory_allocated()/1e6)

                    # if not skip_save:
                    print("saving image")
                    for x_sample in x_samples_ddim:
                        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                        Image.fromarray(x_sample.astype(np.uint8)).save(
                            os.path.join(sample_path, f"{base_count:05}.png"))
                        base_count += 1

                    if not skip_grid:
                        all_samples.append(x_samples_ddim)

            # if not skip_grid:
            #     # additionally, save as grid
            #     grid = torch.stack(all_samples, 0)
            #     grid = rearrange(grid, 'n b c h w -> (n b) c h w')
            #     grid = make_grid(grid, nrow=n_rows)

            #     # to image
            #     grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
            #     Image.fromarray(grid.astype(np.uint8)).save(os.path.join(outpath, f'grid-{grid_count:04}.png'))
            #     grid_count += 1

            # toc = time.time()

print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
        f" \nEnjoy.")

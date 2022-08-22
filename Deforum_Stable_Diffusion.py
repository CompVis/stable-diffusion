# %%
# !! {"metadata":{
# !!   "id": "c442uQJ_gUgy"
# !! }}
"""
# **Deforum Stable Diffusion**
[Stable Diffusion](https://github.com/CompVis/stable-diffusion) by Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, Björn Ommer and the [Stability.ai](https://stability.ai/) Team

Notebook by [deforum](https://twitter.com/deforum_art)
"""

# %%
# !! {"metadata":{
# !!   "id": "2g-f7cQmf2Nt",
# !!   "cellView": "form"
# !! }}
#@markdown **NVIDIA GPU**
!nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

# %%
# !! {"metadata":{
# !!   "id": "VRNl2mfepEIe",
# !!   "cellView": "form"
# !! }}
#@markdown **Setup Environment**

setup_environment = False #@param {type:"boolean"}

if setup_environment:
  %pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
  %pip install omegaconf==2.1.1 einops==0.3.0 pytorch-lightning==1.4.2 torchmetrics==0.6.0 torchtext==0.2.3 transformers==4.19.2 kornia==0.6
  !git clone https://github.com/deforum/stable-diffusion
  %pip install -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers
  %pip install -e git+https://github.com/openai/CLIP.git@main#egg=clip
  %pip install git+https://github.com/deforum/k-diffusion/
  print("Runtime > Restart Runtime")

# %%
# !! {"metadata":{
# !!   "cellView": "form",
# !!   "id": "81qmVZbrm4uu"
# !! }}
#@markdown **Python Definitions**
import json
from IPython import display

import sys, os
import argparse, glob
import torch
import torch.nn as nn
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange, repeat
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext

sys.path.append('./stable-diffusion/')
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

import k_diffusion as K
import accelerate

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())

def get_output_folder(output_path,batch_folder=None):
    yearMonth = time.strftime('%Y-%m/')
    out_path = output_path+"/"+yearMonth
    if batch_folder != "":
        out_path += batch_folder
        if out_path[-1] != "/":
            out_path += "/"
    os.makedirs(out_path, exist_ok=True)
    return out_path

def load_img(path):
    image = Image.open(path).convert("RGB")
    w, h = image.size
    print(f"loaded input image of size ({w}, {h}) from {path}")
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=Image.LANCZOS)
    image = np.array(image).astype(np.float16) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.

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

def run(params):

    # timestring
    timestring = time.strftime('%Y%m%d%H%M%S')

    # outpath
    os.makedirs(params["outdir"], exist_ok=True)
    outpath = params["outdir"]

    # random seed
    if params["seed"] == -1:
        local_seed = np.random.randint(0,4294967295)
    else:
        local_seed = params["seed"]

    # load settings

    # save/append settings
    if params["save_settings"] and params["filename"] is None:
        filename = f"{timestring}_settings.txt"
        assert not os.path.isfile(f"{outpath}{filename}")
        params["filename"] = f"{timestring}_settings.txt"
        params["batch_seeds"] = [local_seed]
        with open(f"{outpath}{filename}", "w+") as f:
            json.dump(params, f, ensure_ascii=False, indent=4)
    elif params["save_settings"] and params["filename"] is not None:
        filename = params["filename"]
        with open(f"{outpath}{filename}") as f:
            params = json.load(f)
        params["batch_seeds"] += [local_seed]
        with open(f"{outpath}{filename}", "w+") as f:
            json.dump(params, f, ensure_ascii=False, indent=4)

    # batch size
    batch_size = params["n_samples"]
    n_rows = params["n_rows"] if params["n_rows"] > 0 else batch_size

    # make prompts
    if not params["from_file"]:
        prompt = params["prompt"]
        assert prompt is not None
        data = [batch_size * [prompt]]
    else:
        infile = params["from_file"]
        print(f"reading prompts from {infile}")
        with open(infile, "r") as f:
            data = f.read().splitlines()
            data = list(chunk(data, batch_size))

    # run
    precision_scope = autocast if params["precision"]=="autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():

                # start timer
                tic = time.time()

                # start
                for prompts in data:

                    # conditional/unconditional
                    uc = None
                    if params["scale"] != 1.0:
                        uc = model.get_learned_conditioning(batch_size * [""])
                    if isinstance(prompts, tuple):
                        prompts = list(prompts)
                    c = model.get_learned_conditioning(prompts)

                    # shape
                    shape = [params["C"], params["H"] // params["f"], params["W"] // params["f"]]

                    # device
                    accelerator = accelerate.Accelerator()
                    device = accelerator.device
                    seeds = torch.randint(-2 ** 63, 2 ** 63 - 1, [accelerator.num_processes])
                    torch.manual_seed(seeds[accelerator.process_index].item())

                    # seed
                    seed_everything(local_seed)

                    # k samplers
                    if params["sampler"] in ["klms","dpm2","dpm2_ancestral","heun","euler","euler_ancestral"]:

                        model_wrap = K.external.CompVisDenoiser(model)
                        sigma_min, sigma_max = model_wrap.sigmas[0].item(), model_wrap.sigmas[-1].item()

                        sigmas = model_wrap.get_sigmas(params["steps"])
                        torch.manual_seed(local_seed)
                        x = torch.randn([params["n_samples"], *shape], device=device) * sigmas[0]
                        model_wrap_cfg = CFGDenoiser(model_wrap)
                        extra_args = {'cond': c, 'uncond': uc, 'cond_scale': params["scale"]}

                        if params["sampler"]=="klms":
                            samples = K.sampling.sample_lms(model_wrap_cfg, x, sigmas, extra_args=extra_args, disable=not accelerator.is_main_process)
                        elif params["sampler"]=="dpm2":
                            samples = K.sampling.sample_dpm_2(model_wrap_cfg, x, sigmas, extra_args=extra_args, disable=not accelerator.is_main_process)
                        elif params["sampler"]=="dpm2_ancestral":
                            samples = K.sampling.sample_dpm_2_ancestral(model_wrap_cfg, x, sigmas, extra_args=extra_args, disable=not accelerator.is_main_process)
                        elif params["sampler"]=="heun":
                            samples = K.sampling.sample_heun(model_wrap_cfg, x, sigmas, extra_args=extra_args, disable=not accelerator.is_main_process)
                        elif params["sampler"]=="euler":
                            samples = K.sampling.sample_euler(model_wrap_cfg, x, sigmas, extra_args=extra_args, disable=not accelerator.is_main_process)
                        elif params["sampler"]=="euler_ancestral":
                            samples = K.sampling.sample_euler_ancestral(model_wrap_cfg, x, sigmas, extra_args=extra_args, disable=not accelerator.is_main_process)
                        
                        x_samples = model.decode_first_stage(samples)
                        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
                        x_samples = accelerator.gather(x_samples)

                    # sd samplers
                    if params["sampler"] in ["plms","ddim"]:

                        # make samplers
                        if params["sampler"]=="plms":
                            params["eta"] = 0
                            sampler = PLMSSampler(model)
                        else:
                            sampler = DDIMSampler(model)

                        # initial image
                        start_code = None
                        if params["use_init"]:

                            assert os.path.isfile(params["init_image"])
                            init_image = load_img(params["init_image"]).to(device)
                            init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
                            init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))  # move to latent space
                            sampler.make_schedule(ddim_num_steps=params['steps'], ddim_eta=params['eta'], verbose=False)
                            assert 0. <= params['strength'] <= 1., 'can only work with strength in [0.0, 1.0]'
                            t_enc = int(params['strength'] * params['steps'])
                            print(f"target t_enc is {t_enc} steps")

                            # encode (scaled latent)
                            z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc]*batch_size).to(device))
                            # decode it
                            samples = sampler.decode(z_enc, c, t_enc,
                                                     unconditional_guidance_scale=params['scale'],
                                                     unconditional_conditioning=uc)
                            x_samples = model.decode_first_stage(samples)
                            x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                        else:

                            if params["fixed_code"]:
                                start_code = torch.randn([params["n_samples"], params["C"], params["H"] // params["f"], params["W"] // params["f"]], device=device)
                            samples, _ = sampler.sample(S=params["steps"],
                                                             conditioning=c,
                                                             batch_size=params["n_samples"],
                                                             shape=shape,
                                                             verbose=False,
                                                             unconditional_guidance_scale=params["scale"],
                                                             unconditional_conditioning=uc,
                                                             eta=params["eta"],
                                                             x_T=start_code)

                            x_samples = model.decode_first_stage(samples)
                            x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
                            x_samples = x_samples


                    # save samples
                    if params["display_samples"] or params["save_samples"]:
                        for count, x_sample in enumerate(x_samples):
                            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                            if params["display_samples"]:
                                display.display(Image.fromarray(x_sample.astype(np.uint8)))
                            if params["save_samples"]:
                                Image.fromarray(x_sample.astype(np.uint8)).save(
                                    os.path.join(outpath, f"{timestring}_{count:02}_{local_seed}.png"))

                    # save grid
                    if params["display_grid"] or params["save_grid"]:
                        grid = torch.stack([x_samples], 0)
                        grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                        grid = make_grid(grid, nrow=n_rows)
                        grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                        if params["display_grid"]:
                            display.display(Image.fromarray(grid.astype(np.uint8)))
                        if params["save_grid"]:
                            Image.fromarray(grid.astype(np.uint8)).save(os.path.join(outpath, f'{timestring}_{local_seed}_grid.png'))
                
                # stop timer
                toc = time.time()                       


# %%
# !! {"metadata":{
# !!   "id": "CzU1bmrigJJB",
# !!   "cellView": "form"
# !! }}
#@markdown **Local Path Variables**
print("Local Path Variables:\n")

models_path = "/content/models" #@param {type:"string"}
output_path = "/content/output" #@param {type:"string"}

#@markdown **Google Drive Path Variables (Optional)**
mount_google_drive = True #@param {type:"boolean"}
force_remount = False

if mount_google_drive:
  from google.colab import drive
  try:
    drive_path = "/content/drive"
    drive.mount(drive_path,force_remount=force_remount)
    models_path = "/content/drive/MyDrive/AI/models" #@param {type:"string"}
    output_path = "/content/drive/MyDrive/AI/StableDiffusion" #@param {type:"string"}
  except:
    print("...error mounting drive or with drive path variables")
    print("...reverting to default path variables")
    models_path = "/content/models"
    output_path = "/content/output"

!mkdir -p $models_path
!mkdir -p $output_path

print(f"models_path: {models_path}")
print(f"output_path: {output_path}")

# -----------------------------------------------------------------------------
#@markdown **Select Model**
print("\nSelect Model:\n")

model_config = "v1-inference.yaml" #@param ["v1-inference.yaml","custom"]
model_checkpoint =  "sd-v1-3-full-ema.ckpt" #@param ["sd-v1-3-full-ema.ckpt","custom"]
check_sha256 = True #@param {type:"boolean"}

model_map = {
    'sd-v1-3-full-ema.ckpt': {'downloaded': False, 'sha256': '54632c6e8a36eecae65e36cb0595fab314e1a1545a65209f24fde221a8d4b2ca', 'link': ['https://drinkordiecdn.lol/sd-v1-3-full-ema.ckpt'] },
  }

def download_model(model_checkpoint):
  download_link = model_map[model_checkpoint]["link"][0]
  print(f"!wget -O {models_path}/{model_checkpoint} {download_link}")
  !wget -O $models_path/$model_checkpoint $download_link
  return

# config path
if os.path.exists(models_path+'/'+model_config):
  print(f"{models_path+'/'+model_config} exists")
else:
  print("cp ./stable-diffusion/configs/stable-diffusion/v1-inference.yaml $models_path/.")
  !cp ./stable-diffusion/configs/stable-diffusion/v1-inference.yaml $models_path/.

# checkpoint path or download
if os.path.exists(models_path+'/'+model_checkpoint):
  print(f"{models_path+'/'+model_checkpoint} exists")
else:
  print("...downloading checkpoint")
  download_model(model_checkpoint)

if check_sha256:
  import hashlib
  print("...checking sha256")
  with open(models_path+'/'+model_checkpoint, "rb") as f:
    bytes = f.read() 
    hash = hashlib.sha256(bytes).hexdigest()
    del bytes
  assert model_map[model_checkpoint]["sha256"] == hash

config = models_path+'/'+model_config
ckpt = models_path+'/'+model_checkpoint

print(f"config: {config}")
print(f"ckpt: {ckpt}")

# -----------------------------------------------------------------------------
#@markdown **Load Model**
print("\nLoad Model:\n")

def load_model_from_config(config, ckpt, verbose=False, device='cuda'):
    map_location = "cuda" #@param ["cpu", "cuda"]
    print(f"...loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location=map_location)
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    #model.cuda()
    model = model.half().to(device)
    model.eval()
    return model

local_config = OmegaConf.load(f"{config}")
model = load_model_from_config(local_config, f"{ckpt}")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = model.to(device)

# %%
# !! {"metadata":{
# !!   "id": "ov3r4RD1tzsT"
# !! }}
"""
# **Run**
"""

# %%
# !! {"metadata":{
# !!   "cellView": "form",
# !!   "id": "qH74gBWDd2oq"
# !! }}
def opt_params():
  
    #@markdown **Save & Display Settings**
    batchdir = "test" #@param {type:"string"}
    outdir = get_output_folder(output_path,batchdir)
    save_settings = False #@param {type:"boolean"}
    save_grid = True #@param {type:"boolean"}
    display_grid = True #@param {type:"boolean"}
    save_samples = True #@param {type:"boolean"}
    display_samples = False #@param {type:"boolean"}

    #@markdown **Prompt Settings**
    seed = 1574552011 #@param
    prompt = "photorealistic painting portrait of a beautiful gorgeous glorious majestic young punjabi princess with headphones figurative liminal complex flat natural realism minimalism by kehinde wiley shadi ghadirian jimmy nelson oil on canvas cosmic levels shimmer pastel color " #@param {type:"string"}
    from_file = False #@param {type:"boolean"}

    #@markdown **Image Settings**
    n_samples = 1 #@param
    n_rows = 1 #@param
    W = 512 #@param
    H = 768 #@param

    #@markdown **Init Settings**
    use_init = True #@param {type:"boolean"}
    init_image = "/content/drive/MyDrive/AI/StableDiffusion/20220815180851_0.png" #@param {type:"string"}
    strength = 0.1 #@param {type:"number"}
    
    #@markdown **Sampling Settings**
    sampler = 'dpm2' #@param ["klms","dpm2","dpm2_ancestral","heun","euler","euler_ancestral","plms", "ddim"]
    steps = 10 #@param
    scale = 7 #@param
    eta = 0.0 #@param
    
    #@markdown **Batch Settings**
    n_batch = 1 #@param

    n_iter = 1
    precision = 'autocast' 
    fixed_code = True
    C = 4
    f = 8

    return locals()

# %%
# !! {"metadata":{
# !!   "id": "cxx8BzxjiaXg",
# !!   "cellView": "form"
# !! }}
#@markdown **Run**
params = opt_params()
params["filename"] = None
for ii in range(params["n_batch"]):
  num = params["n_batch"]
  print(f"run {ii+1} of {num}")
  run(params)

# %%
# !! {"main_metadata":{
# !!   "accelerator": "GPU",
# !!   "colab": {
# !!     "collapsed_sections": [],
# !!     "name": "Deforum Stable Diffusion",
# !!     "provenance": [],
# !!     "private_outputs": true
# !!   },
# !!   "gpuClass": "standard",
# !!   "kernelspec": {
# !!     "display_name": "Python 3",
# !!     "name": "python3"
# !!   },
# !!   "language_info": {
# !!     "name": "python"
# !!   }
# !! }}

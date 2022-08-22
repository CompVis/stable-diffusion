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
import subprocess
sub_p_res = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.free', '--format=csv,noheader'], stdout=subprocess.PIPE).stdout.decode('utf-8')
print(sub_p_res)


# %%
# !! {"metadata":{
# !!   "id": "VRNl2mfepEIe",
# !!   "cellView": "form"
# !! }}
#@markdown **Setup Environment**

setup_environment = False #@param {type:"boolean"}

if setup_environment:
    pip_sub_p_res = subprocess.run(['pip', 'install', 'torch==1.11.0+cu113', 'torchvision==0.12.0+cu113', 'torchaudio==0.11.0', '--extra-index-url', 'https://download.pytorch.org/whl/cu113'], stdout=subprocess.PIPE).stdout.decode('utf-8')
    print(pip_sub_p_res)
    pip_sub_p_res = subprocess.run(['pip', 'install', 'omegaconf==2.1.1', 'einops==0.3.0', 'pytorch-lightning==1.4.2', 'torchmetrics==0.6.0', 'torchtext==0.2.3', 'transformers==4.19.2', 'kornia==0.6'], stdout=subprocess.PIPE).stdout.decode('utf-8')
    print(pip_sub_p_res)
    sub_p_res = subprocess.run(['git', 'clone', 'https://github.com/deforum/stable-diffusion'], stdout=subprocess.PIPE).stdout.decode('utf-8')
    print(sub_p_res)
    pip_sub_p_res = subprocess.run(['pip', 'install', '-e', 'git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers'], stdout=subprocess.PIPE).stdout.decode('utf-8')
    print(pip_sub_p_res)
    pip_sub_p_res = subprocess.run(['pip', 'install', '-e', 'git+https://github.com/openai/CLIP.git@main#egg=clip'], stdout=subprocess.PIPE).stdout.decode('utf-8')
    print(pip_sub_p_res)
    pip_sub_p_res = subprocess.run(['pip', 'install', 'git+https://github.com/deforum/k-diffusion/'], stdout=subprocess.PIPE).stdout.decode('utf-8')
    print(pip_sub_p_res)
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
import shutil
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

def make_callback(sampler, dynamic_threshold=None, static_threshold=None):  
    # Creates the callback function to be passed into the samplers
    # The callback function is applied to the image after each step
    def dynamic_thresholding_(img, threshold):
        # Dynamic thresholding from Imagen paper (May 2022)
        s = np.percentile(np.abs(img.cpu()), threshold, axis=tuple(range(1,img.ndim)))
        s = np.max(np.append(s,1.0))
        torch.clamp_(img, -1*s, s)
        torch.FloatTensor.div_(img, s)

    # Callback for samplers in the k-diffusion repo, called thus:
    #   callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
    def k_callback(args_dict):
        if static_threshold is not None:
            torch.clamp_(args_dict['x'], -1*static_threshold, static_threshold)
        if dynamic_threshold is not None:
            dynamic_thresholding_(args_dict['x'], dynamic_threshold)

    # Function that is called on the image (img) and step (i) at each step
    def img_callback(img, i):
        # Thresholding functions
        if dynamic_threshold is not None:
            dynamic_thresholding_(img, dynamic_threshold)
        if static_threshold is not None:
            torch.clamp_(img, -1*static_threshold, static_threshold)

    if sampler in ["plms","ddim"]: 
        # Callback function formated for compvis latent diffusion samplers
        callback = img_callback
    else: 
        # Default callback function uses k-diffusion sampler variables
        callback = k_callback

    return callback

def run(args, local_seed):

    # load settings
    accelerator = accelerate.Accelerator()
    device = accelerator.device
    seeds = torch.randint(-2 ** 63, 2 ** 63 - 1, [accelerator.num_processes])
    torch.manual_seed(seeds[accelerator.process_index].item())

    # seed
    seed_everything(local_seed)

    # plms
    if args.sampler=="plms":
        args.eta = 0
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    model_wrap = K.external.CompVisDenoiser(model)
    sigma_min, sigma_max = model_wrap.sigmas[0].item(), model_wrap.sigmas[-1].item()

    batch_size = args.n_samples
    n_rows = args.n_rows if args.n_rows > 0 else batch_size

    print(args.prompts)

    data = list(chunk(args.prompts, batch_size))
    sample_index = 0

    start_code = None
    
    # init image
    if args.use_init:
        assert os.path.isfile(args.init_image)
        init_image = load_img(args.init_image).to(device)
        init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
        init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))  # move to latent space

        sampler.make_schedule(ddim_num_steps=args.steps, ddim_eta=args.eta, verbose=False)

        assert 0. <= args.strength <= 1., 'can only work with strength in [0.0, 1.0]'
        t_enc = int(args.strength * args.steps)
        print(f"target t_enc is {t_enc} steps")

    # no init image
    else:
        if args.fixed_code:
            start_code = torch.randn([args.n_samples, args.C, args.H // args.f, args.W // args.f], device=device)

    precision_scope = autocast if args.precision=="autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                tic = time.time()
                for prompt_index, prompts in enumerate(data):
                    prompt_seed = local_seed + prompt_index
                    seed_everything(prompt_seed)

                    callback = make_callback(sampler=args.sampler,
                                            dynamic_threshold=args.dynamic_threshold, 
                                            static_threshold=args.static_threshold)                            

                    uc = None
                    if args.scale != 1.0:
                        uc = model.get_learned_conditioning(batch_size * [""])
                    if isinstance(prompts, tuple):
                        prompts = list(prompts)
                    c = model.get_learned_conditioning(prompts)

                    if args.sampler in ["klms","dpm2","dpm2_ancestral","heun","euler","euler_ancestral"]:
                        shape = [args.C, args.H // args.f, args.W // args.f]
                        sigmas = model_wrap.get_sigmas(args.steps)
                        torch.manual_seed(local_seed)
                        x = torch.randn([args.n_samples, *shape], device=device) * sigmas[0]
                        model_wrap_cfg = CFGDenoiser(model_wrap)
                        extra_args = {'cond': c, 'uncond': uc, 'cond_scale': args.scale}
                        if args.sampler=="klms":
                            samples = K.sampling.sample_lms(model_wrap_cfg, x, sigmas, extra_args=extra_args, disable=not accelerator.is_main_process, callback=callback)
                        elif args.sampler=="dpm2":
                            samples = K.sampling.sample_dpm_2(model_wrap_cfg, x, sigmas, extra_args=extra_args, disable=not accelerator.is_main_process, callback=callback)
                        elif args.sampler=="dpm2_ancestral":
                            samples = K.sampling.sample_dpm_2_ancestral(model_wrap_cfg, x, sigmas, extra_args=extra_args, disable=not accelerator.is_main_process, callback=callback)
                        elif args.sampler=="heun":
                            samples = K.sampling.sample_heun(model_wrap_cfg, x, sigmas, extra_args=extra_args, disable=not accelerator.is_main_process, callback=callback)
                        elif args.sampler=="euler":
                            samples = K.sampling.sample_euler(model_wrap_cfg, x, sigmas, extra_args=extra_args, disable=not accelerator.is_main_process, callback=callback)
                        elif args.sampler=="euler_ancestral":
                            samples = K.sampling.sample_euler_ancestral(model_wrap_cfg, x, sigmas, extra_args=extra_args, disable=not accelerator.is_main_process, callback=callback)

                        x_samples = model.decode_first_stage(samples)
                        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
                        x_samples = accelerator.gather(x_samples)

                    else:

                        # no init image
                        if not args.use_init:
                            shape = [args.C, args.H // args.f, args.W // args.f]

                            samples, _ = sampler.sample(S=args.steps,
                                                            conditioning=c,
                                                            batch_size=args.n_samples,
                                                            shape=shape,
                                                            verbose=False,
                                                            unconditional_guidance_scale=args.scale,
                                                            unconditional_conditioning=uc,
                                                            eta=args.eta,
                                                            x_T=start_code,
                                                            img_callback=callback)

                            x_samples = model.decode_first_stage(samples)
                            x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                        # init image
                        else:
                            # encode (scaled latent)
                            z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc]*batch_size).to(device))
                            # decode it
                            samples = sampler.decode(z_enc, c, t_enc, unconditional_guidance_scale=args.scale,
                                                    unconditional_conditioning=uc,)

                        x_samples = model.decode_first_stage(samples)
                        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
                    
                    # save samples
                    if args.display_samples or args.save_samples:
                        for index, x_sample in enumerate(x_samples):
                            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                            if args.display_samples:
                                display.display(Image.fromarray(x_sample.astype(np.uint8)))
                            if args.save_samples:
                                Image.fromarray(x_sample.astype(np.uint8)).save(
                                    os.path.join(args.outdir, f"{args.timestring}_{index:02}_{prompt_seed}.png"))                                    

                    # save grid
                    if args.display_grid or args.save_grid:
                        grid = torch.stack([x_samples], 0)
                        grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                        grid = make_grid(grid, nrow=n_rows, padding=0)

                        # to image
                        grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                        if args.display_grid:
                            display.display(Image.fromarray(grid.astype(np.uint8)))
                        if args.save_grid:
                            Image.fromarray(grid.astype(np.uint8)).save(os.path.join(args.outdir, f'{args.timestring}_{prompt_seed}_grid.png'))

                # stop timer
                toc = time.time()

    #print(f"Your samples are ready and waiting for you here: \n{outpath} \n" f" \nEnjoy.")

# %%
# !! {"metadata":{
# !!   "cellView": "form",
# !!   "id": "CIUJ7lWI4v53"
# !! }}
#@markdown **Model Path Variables**
# ask for the link
print("Local Path Variables:\n")

models_path = "/content/models" #@param {type:"string"}
output_path = "/content/output" #@param {type:"string"}

#@markdown **Google Drive Path Variables (Optional)**
mount_google_drive = False #@param {type:"boolean"}
force_remount = False

if mount_google_drive:
    from google.colab import drive
    try:
        drive_path = "/content/drive"
        drive.mount(drive_path,force_remount=force_remount)
        models_path_gdrive = "/content/drive/MyDrive/AI/models" #@param {type:"string"}
        output_path_gdrive = "/content/drive/MyDrive/AI/StableDiffusion" #@param {type:"string"}
        models_path = models_path_gdrive
        output_path = output_path_gdrive
    except:
        print("...error mounting drive or with drive path variables")
        print("...reverting to default path variables")

os.makedirs(models_path, exist_ok=True)
os.makedirs(output_path, exist_ok=True)

print(f"models_path: {models_path}")
print(f"output_path: {output_path}")

# -----------------------------------------------------------------------------
#@markdown **Select Model**
print("\nSelect Model:\n")

model_config = "v1-inference.yaml" #@param ["v1-inference.yaml","custom"]
model_checkpoint =  "sd-v1-3-full-ema.ckpt" #@param ["sd-v1-3-full-ema.ckpt","sd-v1-4.ckpt"]
check_sha256 = True #@param {type:"boolean"}

model_map = {
    'sd-v1-3-full-ema.ckpt': {'downloaded': False, 'sha256': '54632c6e8a36eecae65e36cb0595fab314e1a1545a65209f24fde221a8d4b2ca', 'link': ['https://drinkordiecdn.lol/sd-v1-3-full-ema.ckpt'] },
}

def wget(url, outputdir):
    res = subprocess.run(['wget', url, '-P', f'{outputdir}'], stdout=subprocess.PIPE).stdout.decode('utf-8')
    print(res)

def download_model(model_checkpoint):
    download_link = model_map[model_checkpoint]["link"][0]
    print(f"!wget -O {models_path}/{model_checkpoint} {download_link}")
    wget(download_link, models_path)
    return

# config path
if os.path.exists(models_path+'/'+model_config):
    print(f"{models_path+'/'+model_config} exists")
else:
    print("cp ./stable-diffusion/configs/stable-diffusion/v1-inference.yaml $models_path/.")
    shutil.copy('./stable-diffusion/configs/stable-diffusion/v1-inference.yaml', models_path)

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


# %%
# !! {"metadata":{
# !!   "cellView": "form",
# !!   "id": "IJiMgz_96nr3"
# !! }}
#@markdown **Load Stable Diffusion**

def load_model_from_config(config, ckpt, verbose=False, device='cuda'):
    map_location = "cuda" #@param ["cpu", "cuda"]
    print(f"Loading model from {ckpt}")
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
class DeforumArgs():
    def __init__(self):

        #@markdown **Save & Display Settings**
        self.batchdir = "test" #@param {type:"string"}
        self.outdir = get_output_folder(output_path, self.batchdir)
        self.save_settings = False #@param {type:"boolean"}
        self.save_grid = True #@param {type:"boolean"}
        self.display_grid = True #@param {type:"boolean"}
        self.save_samples = True #@param {type:"boolean"}
        self.display_samples = False #@param {type:"boolean"}

        #@markdown **Prompt Settings**
        self.seed = 1574552011 #@param

        #@markdown **Image Settings**
        self.n_samples = 1 #@param
        self.n_rows = 1 #@param
        self.W = 512 #@param
        self.H = 768 #@param

        #@markdown **Init Settings**
        self.use_init = False #@param {type:"boolean"}
        self.init_image = "/content/drive/MyDrive/AI/StableDiffusion/20220815180851_0.png" #@param {type:"string"}
        self.strength = 0.1 #@param {type:"number"}

        #@markdown **Sampling Settings**
        self.sampler = 'dpm2' #@param ["klms","dpm2","dpm2_ancestral","heun","euler","euler_ancestral","plms", "ddim"]
        self.steps = 10 #@param
        self.scale = 7 #@param
        self.eta = 0.0 #@param
        self.dynamic_threshold = None #@param
        self.static_threshold = None #@param    

        #@markdown **Batch Settings**
        self.n_batch = 1 #@param

        self.precision = 'autocast' 
        self.fixed_code = True
        self.C = 4
        self.f = 8
        self.prompts = prompts
        self.timestring = ""

# %%
# !! {"metadata":{
# !!   "id": "2ujwkGZTcGev"
# !! }}
prompts = [
    "a beautiful forest by Asher Brown Durand, trending on Artstation", #the first prompt I want
    "a beautiful portrait of a woman by Artgerm, trending on Artstation", #the second prompt I want
    #["the third prompt I don't want it I commented it with an #"],
]

# %%
# !! {"metadata":{
# !!   "cellView": "form",
# !!   "id": "cxx8BzxjiaXg"
# !! }}
#@markdown **Run**
args = DeforumArgs()
args.filename = None
args.prompts = prompts

def do_batch_run():
    # create output folder
    os.makedirs(args.outdir, exist_ok=True)

    # current timestring for filenames
    args.timestring = time.strftime('%Y%m%d%H%M%S')

    # random seed
    if args.seed == -1:
        local_seed = np.random.randint(0,4294967295)
    else:
        local_seed = args.seed

    # save settings for the batch
    if args.save_settings:
        filename = os.path.join(args.outdir, f"{args.timestring}_settings.txt")
        with open(filename, "w+", encoding="utf-8") as f:
            json.dump(dict(args.__dict__), f, ensure_ascii=False, indent=4)

    for batch_index in range(args.n_batch):
        print(f"run {batch_index+1} of {args.n_batch}")
        run(args, local_seed)
        local_seed += 1

do_batch_run()

# %%
# !! {"main_metadata":{
# !!   "accelerator": "GPU",
# !!   "colab": {
# !!     "collapsed_sections": [],
# !!     "name": "Deforum_Stable_Diffusion.ipynb",
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
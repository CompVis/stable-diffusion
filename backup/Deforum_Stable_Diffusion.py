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

setup_environment = True #@param {type:"boolean"}
print_subprocess = False #@param {type:"boolean"}

if setup_environment:
    import subprocess
    print("...setting up environment")
    all_process = [['pip', 'install', 'torch==1.11.0+cu113', 'torchvision==0.12.0+cu113', 'torchaudio==0.11.0', '--extra-index-url', 'https://download.pytorch.org/whl/cu113'],
                   ['pip', 'install', 'omegaconf==2.1.1', 'einops==0.3.0', 'pytorch-lightning==1.4.2', 'torchmetrics==0.6.0', 'torchtext==0.2.3', 'transformers==4.19.2', 'kornia==0.6'],
                   ['git', 'clone', 'https://github.com/deforum/stable-diffusion'],
                   ['pip', 'install', '-e', 'git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers'],
                   ['pip', 'install', '-e', 'git+https://github.com/openai/CLIP.git@main#egg=clip'],
                   ['pip', 'install', 'accelerate', 'ftfy', 'jsonmerge', 'resize-right', 'torchdiffeq'],
                 ]
    for process in all_process:
        running = subprocess.run(process,stdout=subprocess.PIPE).stdout.decode('utf-8')
        if print_subprocess:
            print(running)
    
    print(subprocess.run(['git', 'clone', 'https://github.com/deforum/k-diffusion/'], stdout=subprocess.PIPE).stdout.decode('utf-8'))
    with open('k-diffusion/k_diffusion/__init__.py', 'w') as f:
        f.write('')
    
    import sys
    sys.path.append('./src/taming-transformers')
    sys.path.append('./src/clip')
    sys.path.append('./stable-diffusion/')
    sys.path.append('./k-diffusion')

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
import requests
import shutil
from types import SimpleNamespace
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

from helpers import save_samples
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

import accelerate
from k_diffusion import sampling
from k_diffusion.external import CompVisDenoiser

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

def load_img(path, shape):
    if path.startswith('http://') or path.startswith('https://'):
        image = Image.open(requests.get(path, stream=True).raw).convert('RGB')
    else:
        image = Image.open(path).convert('RGB')

    image = image.resize(shape, resample=Image.LANCZOS)
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

    # plms
    if args.sampler=="plms":
        args.eta = 0
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    model_wrap = CompVisDenoiser(model)
    sigma_min, sigma_max = model_wrap.sigmas[0].item(), model_wrap.sigmas[-1].item()

    batch_size = args.n_samples
    n_rows = args.n_rows if args.n_rows > 0 else batch_size

    data = list(chunk(args.prompts, batch_size))
    sample_index = 0

    start_code = None
    
    # init image
    if args.use_init:
        assert os.path.isfile(args.init_image)
        init_image = load_img(args.init_image, shape=(args.W, args.H)).to(device)
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
                    print(prompts)
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
                        torch.manual_seed(prompt_seed)
                        if args.use_init:
                            sigmas = sigmas[t_enc:]
                            x = init_latent + torch.randn([args.n_samples, *shape], device=device) * sigmas[0]
                        else:
                            x = torch.randn([args.n_samples, *shape], device=device) * sigmas[0]
                        model_wrap_cfg = CFGDenoiser(model_wrap)
                        extra_args = {'cond': c, 'uncond': uc, 'cond_scale': args.scale}
                        if args.sampler=="klms":
                            samples = sampling.sample_lms(model_wrap_cfg, x, sigmas, extra_args=extra_args, disable=not accelerator.is_main_process, callback=callback)
                        elif args.sampler=="dpm2":
                            samples = sampling.sample_dpm_2(model_wrap_cfg, x, sigmas, extra_args=extra_args, disable=not accelerator.is_main_process, callback=callback)
                        elif args.sampler=="dpm2_ancestral":
                            samples = sampling.sample_dpm_2_ancestral(model_wrap_cfg, x, sigmas, extra_args=extra_args, disable=not accelerator.is_main_process, callback=callback)
                        elif args.sampler=="heun":
                            samples = sampling.sample_heun(model_wrap_cfg, x, sigmas, extra_args=extra_args, disable=not accelerator.is_main_process, callback=callback)
                        elif args.sampler=="euler":
                            samples = sampling.sample_euler(model_wrap_cfg, x, sigmas, extra_args=extra_args, disable=not accelerator.is_main_process, callback=callback)
                        elif args.sampler=="euler_ancestral":
                            samples = sampling.sample_euler_ancestral(model_wrap_cfg, x, sigmas, extra_args=extra_args, disable=not accelerator.is_main_process, callback=callback)

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

                        # init image
                        else:
                            # encode (scaled latent)
                            z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc]*batch_size).to(device))
                            # decode it
                            samples = sampler.decode(z_enc, c, t_enc, unconditional_guidance_scale=args.scale,
                                                    unconditional_conditioning=uc,)

                        x_samples = model.decode_first_stage(samples)
                        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
                    

                    grid, images = save_samples(
                        args, x_samples=x_samples, seed=prompt_seed, n_rows=n_rows
                    )
                    if args.display_samples:
                        for im in images:
                            display.display(im)
                    if args.display_grid:
                        display.display(grid)

                # stop timer
                toc = time.time()

    #print(f"Your samples are ready and waiting for you here: \n{outpath} \n" f" \nEnjoy.")

# %%
# !! {"metadata":{
# !!   "cellView": "form",
# !!   "id": "TxIOPT0G5Lx1"
# !! }}
#@markdown **Model Path Variables**
# ask for the link
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

# %%
# !! {"metadata":{
# !!   "cellView": "form",
# !!   "id": "CIUJ7lWI4v53"
# !! }}
#@markdown **Select Model**
print("\nSelect Model:\n")

model_config = "v1-inference.yaml" #@param ["custom","v1-inference.yaml"]
model_checkpoint =  "sd-v1-4.ckpt" #@param ["custom","sd-v1-4.ckpt","sd-v1-3-full-ema.ckpt","sd-v1-3.ckpt","sd-v1-2-full-ema.ckpt","sd-v1-2.ckpt","sd-v1-1-full-ema.ckpt","sd-v1-1.ckpt"]
custom_config_path = "" #@param {type:"string"}
custom_checkpoint_path = "" #@param {type:"string"}

check_sha256 = True #@param {type:"boolean"}

model_map = {
    "sd-v1-4.ckpt": {'sha256': 'fe4efff1e174c627256e44ec2991ba279b3816e364b49f9be2abc0b3ff3f8556'},
    "sd-v1-3-full-ema.ckpt": {'sha256': '54632c6e8a36eecae65e36cb0595fab314e1a1545a65209f24fde221a8d4b2ca'},
    "sd-v1-3.ckpt": {'sha256': '2cff93af4dcc07c3e03110205988ff98481e86539c51a8098d4f2236e41f7f2f'},
    "sd-v1-2-full-ema.ckpt": {'sha256': 'bc5086a904d7b9d13d2a7bccf38f089824755be7261c7399d92e555e1e9ac69a'},
    "sd-v1-2.ckpt": {'sha256': '3b87d30facd5bafca1cbed71cfb86648aad75d1c264663c0cc78c7aea8daec0d'},
    "sd-v1-1-full-ema.ckpt": {'sha256': 'efdeb5dc418a025d9a8cc0a8617e106c69044bc2925abecc8a254b2910d69829'},
    "sd-v1-1.ckpt": {'sha256': '86cd1d3ccb044d7ba8db743d717c9bac603c4043508ad2571383f954390f3cea'}
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
    print(f"download model checkpoint and place in {models_path+'/'+model_checkpoint}")
    #download_model(model_checkpoint)

if check_sha256:
    import hashlib
    print("\n...checking sha256")
    with open(models_path+'/'+model_checkpoint, "rb") as f:
        bytes = f.read() 
        hash = hashlib.sha256(bytes).hexdigest()
        del bytes
    if model_map[model_checkpoint]["sha256"] == hash:
        print("hash is correct\n")
    else:
        print("hash in not correct\n")

if model_config == "custom":
  config = custom_config_path
else:
  config = models_path+'/'+model_config

if model_checkpoint == "custom":
  ckpt = custom_checkpoint_path
else:
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

load_on_run_all = True #@param {type: 'boolean'}

if load_on_run_all:

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
# !!   "id": "qH74gBWDd2oq"
# !! }}
def DeforumArgs():
    #@markdown **Save & Display Settings**
    batchdir = "test" #@param {type:"string"}
    outdir = get_output_folder(output_path, batchdir)
    save_grid = False
    save_samples = True #@param {type:"boolean"}
    save_settings = True #@param {type:"boolean"}
    display_grid = False
    display_samples = True #@param {type:"boolean"}

    #@markdown **Image Settings**
    n_samples = 1 #@param
    n_rows = 1 #@param
    W = 512 #@param
    H = 576 #@param
    W, H = map(lambda x: x - x % 64, (W, H))  # resize to integer multiple of 64


    #@markdown **Init Settings**
    use_init = False #@param {type:"boolean"}
    init_image = "/content/drive/MyDrive/AI/escape.jpg" #@param {type:"string"}
    strength = 0.5 #@param {type:"number"}

    #@markdown **Sampling Settings**
    seed = 1 #@param
    sampler = 'euler_ancestral' #@param ["klms","dpm2","dpm2_ancestral","heun","euler","euler_ancestral","plms", "ddim"]
    steps = 50 #@param
    scale = 7 #@param
    eta = 0.0 #@param
    dynamic_threshold = None
    static_threshold = None   

    #@markdown **Batch Settings**
    n_batch = 2 #@param

    precision = 'autocast' 
    fixed_code = True
    C = 4
    f = 8
    prompts = []
    timestring = ""

    return locals()

args = SimpleNamespace(**DeforumArgs())


# %%
# !! {"metadata":{
# !!   "id": "2ujwkGZTcGev"
# !! }}
prompts = [
    "a beautiful forest by Asher Brown Durand, trending on Artstation", #the first prompt I want
    "a beautiful portrait of a woman by Artgerm, trending on Artstation", #the second prompt I want
    #"the third prompt I don't want it I commented it with an",
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

    # save settings for the batch
    if args.save_settings:
        filename = os.path.join(args.outdir, f"{args.timestring}_settings.txt")
        with open(filename, "w+", encoding="utf-8") as f:
            json.dump(dict(args.__dict__), f, ensure_ascii=False, indent=4)

    for batch_index in range(args.n_batch):

        # random seed
        if args.seed == -1:
            local_seed = np.random.randint(0,4294967295)
        else:
            local_seed = args.seed

        print(f"run {batch_index+1} of {args.n_batch}")
        run(args, local_seed)

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

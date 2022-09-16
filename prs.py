import argparse, os, sys, glob
import random
import shutil
import torch
import re
from torch import nn
from torch import Tensor
import numpy as np
from omegaconf import OmegaConf
import PIL
from PIL import Image, ImageOps, ImageStat, ImageEnhance, ImageDraw
from PIL.PngImagePlugin import PngInfo
from einops import rearrange, repeat
from tqdm import tqdm, trange
from itertools import islice
from typing import Iterable
import time
from pytorch_lightning import seed_everything
from torch import autocast
#import accelerate
from contextlib import contextmanager, nullcontext
import subprocess

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

from k_diffusion.sampling import sample_lms, sample_dpm_2, sample_dpm_2_ancestral, sample_euler, sample_euler_ancestral, sample_heun, get_sigmas_karras, append_zero
from k_diffusion.external import CompVisDenoiser

from types import SimpleNamespace
import json5 as json

try:
    # this silences the annoying "Some weights of the model checkpoint were not used when initializing..." message at start.
    from transformers import logging
    logging.set_verbosity_error()
except:
    pass

# samplers from the Karras et al paper
KARRAS_SAMPLERS = { 'k_heun', 'k_euler', 'k_dpm_2' }
NON_KARRAS_K_DIFF_SAMPLERS = { 'k_lms', 'k_dpm_2_ancestral', 'k_euler_ancestral' }
K_DIFF_SAMPLERS = { *KARRAS_SAMPLERS, *NON_KARRAS_K_DIFF_SAMPLERS }
NOT_K_DIFF_SAMPLERS = { 'ddim', 'plms' }
VALID_SAMPLERS = { *K_DIFF_SAMPLERS, *NOT_K_DIFF_SAMPLERS }

class KCFGDenoiser(nn.Module):
    inner_model: CompVisDenoiser
    def __init__(self, model: CompVisDenoiser):
        super().__init__()
        self.inner_model = model

    def forward(self, x: Tensor, sigma: Tensor, uncond: Tensor, conditions: Iterable[Tensor], cond_scale: float) -> Tensor:
        x_in = torch.cat([x] * 2)
        sigma_in = torch.cat([sigma] * 2)
        cond_in = torch.cat([uncond, *conditions])
        conditions_len = len(conditions)
        uncond, *conditions = self.inner_model(x_in, sigma_in, cond=cond_in).chunk(1 + conditions_len)
        cond = torch.sum(torch.stack(conditions), dim=0) / conditions_len
        return uncond + (cond - uncond) * cond_scale

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    # if "global_step" in pl_sd:
    #     print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    return model

def get_resampling_mode():
    try:
        from PIL import __version__, Image
        major_ver = int(__version__.split('.')[0])
        if major_ver >= 9:
            return Image.Resampling.LANCZOS
        else:
            return Image.LANCZOS
    except Exception as ex:
        return 1  # 'Lanczos' irrespective of version.

def load_img(path):
    image = Image.open(path).convert("RGB")
    w, h = image.size
    #print(f"loaded input image of size ({w}, {h}) from {path}")
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), get_resampling_mode())
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.

def thats_numberwang(dir, wildcard):
    # get the highest numbered file in the out directory, and add 1. So simple.
    files = os.listdir(dir)
    filenums = []
    filenum = 0
    for file in files:
        if wildcard in file:
            start = file.rfind('-')
            end = file.rfind('.')
            try:
                filenum = file[start + 1:end]
                filenum = int(filenum)
            except:
                print(f'Improperly named file "{file}" in output directory')
                print(f'Please make sure output filenames use the name-1234.png format')
                quit()
            filenums.append(filenum)
    if not filenums:
        numberwang = 0
    else:
        numberwang = max(filenums) + 1
    return numberwang

def slerp(device, t, v0:torch.Tensor, v1:torch.Tensor, DOT_THRESHOLD=0.9995):
    v0 = v0.detach().cpu().numpy()
    v1 = v1.detach().cpu().numpy()
    
    dot = np.sum(v0 * v1 / (np.linalg.norm(v0) * np.linalg.norm(v1)))
    if np.abs(dot) > DOT_THRESHOLD:
        v2 = (1 - t) * v0 + t * v1
    else:
        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)
        theta_t = theta_0 * t
        sin_theta_t = np.sin(theta_t)
        s0 = np.sin(theta_0 - theta_t) / sin_theta_0
        s1 = sin_theta_t / sin_theta_0
        v2 = s0 * v0 + s1 * v1

    v2 = torch.from_numpy(v2).to(device)

    return v2

def split_weighted_subprompts(input_string, normalize=True):
    parsed_prompts = [(match.group("prompt").replace("\\:", ":"), float(match.group("weight") or 1)) for match in re.finditer(prompt_parser, input_string)]
    if not normalize:
        return parsed_prompts
    weight_sum = sum(map(lambda x: x[1], parsed_prompts))
    if weight_sum == 0:
        print("Warning: Subprompt weights add up to zero. Discarding and using even weights instead.")
        equal_weight = 1 / (len(parsed_prompts) or 1)
        return [(x[0], equal_weight) for x in parsed_prompts]
    return [(x[0], x[1] / weight_sum) for x in parsed_prompts]

prompt_parser = re.compile("""
    (?P<prompt>     # capture group for 'prompt'
    (?:\\\:|[^:])+  # match one or more non ':' characters or escaped colons '\:'
    )               # end 'prompt'
    (?:             # non-capture group
    :+              # match one or more ':' characters
    (?P<weight>     # capture group for 'weight'
    -?\d+(?:\.\d+)? # match positive or negative integer or decimal number
    )?              # end weight capture group, make optional
    \s*             # strip spaces after weight
    |               # OR
    $               # else, if no ':' then match end of line
    )               # end non-capture group
""", re.VERBOSE)

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

def do_run(device, model, opt):
    print(f'Starting render!')
    seed_everything(opt.seed)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    batch_size = 1

    # prompt = opt.prompt
    data = [batch_size * [opt.prompt]]
    # data = opt.prompt

    # grid is a leftover from stable, but we use it to give our output file a unique name
    grid_count = thats_numberwang(outpath, opt.batch_name)

    progress_image = "progress.jpg" if opt.filetype == ".jpg" else "progress.png" 
   
    if opt.method in K_DIFF_SAMPLERS:
        model_k_wrapped = CompVisDenoiser(model, quantize=True)
        model_k_guidance = KCFGDenoiser(model_k_wrapped)
    elif opt.method in NOT_K_DIFF_SAMPLERS:
        if opt.method == 'plms':
            sampler = PLMSSampler(model, device)
        else:
            sampler = DDIMSampler(model, device)

    def img_to_latent(path: str) -> Tensor:
        assert os.path.isfile(path)
        if device.type == "cuda":
            image = load_img(path).to(device).half()
        else:
            image = load_img(path).to(device)
        image = repeat(image, '1 ... -> b ...', b=batch_size)
        latent: Tensor = model.get_first_stage_encoding(model.encode_first_stage(image))  # move to latent space
        return latent
    
    if opt.init_image is not None:
        init_latent = img_to_latent(opt.init_image)
        assert 0. <= opt.strength <= 1., 'can only work with strength in [0.0, 1.0]'
        t_enc = int(opt.strength * opt.ddim_steps)
    else:
        init_latent = None

    shape = [opt.C, opt.H // opt.f, opt.W // opt.f]

    precision_scope = autocast if opt.precision=="autocast" else nullcontext
    # apple silicon support
    if device.type == 'mps':
        precision_scope = nullcontext

    rand_size = [batch_size, *shape]
    og_start_code = torch.randn(rand_size, device='cpu').to(device) if device.type == 'mps' else torch.randn(rand_size, device=device)
    start_code = og_start_code

    with torch.no_grad():
        with precision_scope(device.type):
            with model.ema_scope():
                for n in trange(opt.n_iter, desc="Sampling"):
                    for prompts in tqdm(data, desc="data"):
                        uc = None
                        if opt.scale != 1.0:
                            uc = model.get_learned_conditioning(batch_size * [""])

                        # process the prompt for randomizers and dynamic values
                        newprompts = []
                        for prompt in prompts:
                            prompt = randomize_prompt(prompt)
                            prompt = dynamic_value(prompt)
                            newprompts.append(prompt)
                        prompts = newprompts

                        print(f'\nPrompt for this image:\n   {prompts}\n')
                        # split the prompt if it has : for weighting
                        normalize_prompt_weights = True
                        weighted_subprompts = split_weighted_subprompts(prompts[0], normalize_prompt_weights)

                        # save a settings file for this image
                        if opt.save_settings:
                            save_settings(opt, prompts[0], grid_count)

                        # sub-prompt weighting used if more than 1
                        if len(weighted_subprompts) > 1:
                            c = torch.zeros_like(uc) # i dont know if this is correct.. but it works
                            for i in range(0, len(weighted_subprompts)):
                                # note if alpha negative, it functions same as torch.sub
                                c = torch.add(c, model.get_learned_conditioning(weighted_subprompts[i][0]), alpha=weighted_subprompts[i][1])
                        else: # just behave like usual
                            c = model.get_learned_conditioning(prompts)
                        
                        if opt.variance != 0.0 and n != 0:
                            # add a little extra random noise to get varying output with same seed
                            base_x = og_start_code # torch.randn(rand_size, device=device) * sigmas[0]
                            torch.manual_seed(opt.variance_seed + n)
                            target_x = torch.randn(rand_size, device='cpu').to(device) if device.type == 'mps' else torch.randn(rand_size, device=device)
                            start_code = slerp(device, max(0.0, min(1.0, opt.variance)), base_x, target_x)

                        karras_noise = False

                        if opt.method in NOT_K_DIFF_SAMPLERS:
                            if init_latent is None:
                                samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                                conditioning=c,
                                                                batch_size=batch_size,
                                                                shape=shape,
                                                                verbose=False,
                                                                unconditional_guidance_scale=opt.scale,
                                                                unconditional_conditioning=uc,
                                                                eta=opt.ddim_eta,
                                                                x_T=start_code)
                                sigmas = None
                            else:
                                # encode (scaled latent)
                                z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc]*batch_size).to(device))
                                # decode it
                                samples_ddim = sampler.decode(z_enc, c, t_enc, unconditional_guidance_scale=opt.scale,
                                                        unconditional_conditioning=uc,)

                        else:
                            if opt.method == 'k_dpm_2':
                                sampling_fn = sample_dpm_2
                                karras_noise = True
                            elif opt.method == 'k_dpm_2_ancestral':
                                sampling_fn = sample_dpm_2_ancestral
                            elif opt.method == 'k_heun':
                                sampling_fn = sample_heun
                                karras_noise = True
                            elif opt.method == 'k_euler':
                                sampling_fn = sample_euler
                                karras_noise = True
                            elif opt.method == 'k_euler_ancestral':
                                sampling_fn = sample_euler_ancestral
                            else:
                                sampling_fn = sample_lms

                            noise_schedule_sampler_args = {}

                            if karras_noise:
                                end_karras_ramp_early = False # this is only needed for really low step counts, not going to bother with it right now
                                def get_premature_sigma_min(
                                    steps: int,
                                    sigma_max: float,
                                    sigma_min_nominal: float,
                                    rho: float
                                ) -> float:
                                    min_inv_rho = sigma_min_nominal ** (1 / rho)
                                    max_inv_rho = sigma_max ** (1 / rho)
                                    ramp = (steps-2) * 1/(steps-1)
                                    sigma_min = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
                                    return sigma_min

                                rho = 7.
                                sigma_max=model_k_wrapped.sigmas[-1].item()
                                sigma_min_nominal=model_k_wrapped.sigmas[0].item()
                                premature_sigma_min = get_premature_sigma_min(
                                    steps=opt.ddim_steps+1,
                                    sigma_max=sigma_max,
                                    sigma_min_nominal=sigma_min_nominal,
                                    rho=rho
                                )
                                sigmas = get_sigmas_karras(
                                    n=opt.ddim_steps,
                                    sigma_min=premature_sigma_min if end_karras_ramp_early else sigma_min_nominal,
                                    sigma_max=sigma_max,
                                    rho=rho,
                                    device=device,
                                )

                            else:
                                sigmas = model_k_wrapped.get_sigmas(opt.ddim_steps)
                            
                            if init_latent is not None:
                                sigmas = sigmas[len(sigmas) - t_enc - 1 :]

                            x = start_code * sigmas[0] # for GPU draw
                            if init_latent is not None:
                                x = init_latent + x

                            extra_args = {
                                'conditions': (c,),
                                'uncond': uc,
                                'cond_scale': opt.scale,
                            }
                            samples_ddim = sampling_fn(
                                model_k_guidance,
                                x,
                                sigmas,
                                extra_args=extra_args,
                                **noise_schedule_sampler_args)


                        x_samples_ddim = model.decode_first_stage(samples_ddim)
                        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

                        metadata = PngInfo()
                        if opt.hide_metadata == False:
                            metadata.add_text("prompt", str(prompts))
                            metadata.add_text("seed", str(opt.seed))
                            metadata.add_text("steps", str(opt.ddim_steps))
                            metadata.add_text("scale", str(opt.scale))
                            metadata.add_text("ETA", str(opt.ddim_eta))
                            metadata.add_text("method", str(opt.method))
                            metadata.add_text("init_image", str(opt.init_image))

                        for x_sample in x_samples_ddim:
                            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                            output_filename = os.path.join(outpath, f'{opt.batch_name}{opt.device_id}-{grid_count:04}{opt.filetype}')
                            output_image = Image.fromarray(x_sample.astype(np.uint8))
                            output_image.save(progress_image, pnginfo=metadata, quality = opt.quality)
                            shutil.copy2(progress_image, output_filename)
                            output_image.close()
                            print(f'\nOutput saved as "{output_filename}"\n')
                            grid_count += 1

                toc = time.time()
    return output_filename

#functions for GO BIG
def addalpha(im, mask):
    imr, img, imb, ima = im.split()
    mmr, mmg, mmb, mma = mask.split()
    im = Image.merge('RGBA', [imr, img, imb, mma])  # we want the RGB from the original, but the transparency from the mask
    return(im)

# Alternative method composites a grid of images at the positions provided
def grid_merge(source, slices):
    source.convert("RGBA")
    for slice, posx, posy in slices: # go in reverse to get proper stacking
        source.alpha_composite(slice, (posx, posy))
    return source

def grid_coords(target, original, overlap, maxed):
    #generate a list of coordinate tuples for our sections, in order of how they'll be rendered
    #target should be the size for the gobig result, original is the size of each chunk being rendered
    target_x, target_y = target
    original_x, original_y = original
    do_calc = True
    while do_calc:
        print(f'Target size is {target_x} x {target_y}')
        center = []
        center_x = int(target_x / 2)
        center_y = int(target_y / 2)
        x = center_x - int(original_x / 2)
        y = center_y - int(original_y / 2)
        center.append((x,y)) #center chunk
        uy = y #up
        uy_list = []
        dy = y #down
        dy_list = []
        lx = x #left
        lx_list = []
        rx = x #right
        rx_list = []
        while uy > 0: #center row vertical up
            uy = uy - original_y + overlap
            uy_list.append((lx, uy))
        while (dy + original_y) <= target_y: #center row vertical down
            dy = dy + original_y - overlap
            dy_list.append((rx, dy))
        while lx > 0:
            lx = lx - original_x + overlap
            lx_list.append((lx, y))
            uy = y
            while uy > 0:
                uy = uy - original_y + overlap
                uy_list.append((lx, uy))
            dy = y
            while (dy + original_y) <= target_y:
                dy = dy + original_y - overlap
                dy_list.append((lx, dy))
        while (rx + original_x) <= target_x:
            rx = rx + original_x - overlap
            rx_list.append((rx, y))
            uy = y
            while uy > 0:
                uy = uy - original_y + overlap
                uy_list.append((rx, uy))
            dy = y
            while (dy + original_y) <= target_y:
                dy = dy + original_y - overlap
                dy_list.append((rx, dy))
        if maxed:
            # calculate a new size that will fill the canvas, which will be optionally used in grid_slice and go_big
            last_coordx, last_coordy = dy_list[-1:][0]
            render_edgey = last_coordy + original_y # outer bottom edge of the render canvas
            render_edgex = last_coordx + original_x # outer side edge of the render canvas
            render_edgex += (render_edgex - target_x) # we have to extend the "negative" side as well, so we do it twice
            render_edgey += (render_edgey - target_y)
            scalarx = render_edgex / target_x
            scalary = render_edgey / target_y
            if scalarx <= scalary:
                target_x = int(target_x * scalarx)
                target_y = int(target_y * scalarx)
            else:
                target_x = int(target_x * scalary)
                target_y = int(target_y * scalary)
            maxed = False
        else:
            do_calc = False
    # now put all the chunks into one master list of coordinates (essentially reverse of how we calculated them so that the central slices will be on top)
    result = []
    for coords in dy_list[::-1]:
        result.append(coords)
    for coords in uy_list[::-1]:
        result.append(coords)
    for coords in rx_list[::-1]:
        result.append(coords)
    for coords in lx_list[::-1]:
        result.append(coords)
    result.append(center[0])
    return result, (target_x, target_y)

# Chop our source into a grid of images that each equal the size of the original render
def grid_slice(source, overlap, og_size, maxed=False): 
    width, height = og_size # size of the slices to be rendered
    coordinates, new_size = grid_coords(source.size, og_size, overlap, maxed)
    if source.size != new_size:
        source = source.resize(new_size, get_resampling_mode())
    slices = []
    for coordinate in coordinates:
        x, y = coordinate
        slices.append(((source.crop((x, y, x+width, y+height))), x, y))
    global slices_todo
    slices_todo = len(slices) - 1
    return slices, source

def parse_args():
    my_parser = argparse.ArgumentParser(
        prog='ProgRock-Stable',
        description='Generate images from text prompts, based on Stable Diffusion.',
    )
    my_parser.add_argument(
        '-s',
        '--settings',
        action='append',
        required=False,
        default=['settings.json'],
        help='A settings JSON file to use, best to put in quotes. Multiples are allowed and layered in order.'
    )
    my_parser.add_argument(
        '-o',
        '--output',
        action='store',
        required=False,
        help='What output directory to use within images_out'
    )
    my_parser.add_argument(
        '-p',
        '--prompt',
        type=str,
        action='store',
        required=False,
        help='Override the prompt'
    )
    my_parser.add_argument(
        '-c',
        '--cpu',
        type=int,
        nargs='?',
        action='store',
        required=False,
        default=False,
        const=0,
        help='Force use of CPU instead of GPU, and how many threads to run'
    )
    my_parser.add_argument(
        '-n',
        '--n_batches',
        type=int,
        action='store',
        required=False,
        help='How many batches of images to generate'
    )
    my_parser.add_argument(
        '-i',
        '--n_iter',
        type=int,
        action='store',
        required=False,
        help='How many images to generate within a batch'
    )
    my_parser.add_argument(
        '--seed',
        type=int,
        action='store',
        required=False,
        help='Specify the numeric seed to be used'
    )
    my_parser.add_argument(
        '-f',
        '--from_file',
        action='store',
        required=False,
        help='A text file with prompts (one per line)'
    )
    my_parser.add_argument(
        '--gobig',
        action='store_true',
        required=False,
        help='After generation, the image is split into sections and re-rendered, to double the size.'
    )
    my_parser.add_argument(
        '--gobig_init',
        action='store',
        required=False,
        help='An image to use to kick off GO BIG mode, skipping the initial render.'
    )
    my_parser.add_argument(
        '--gobig_scale',
        action='store',
        type=int,
        default = 2,
        required=False,
        help='What scale to multiply your original image by. 2 is a good value. 3 is insane. Anything more and I wish you luck.'
    )
    my_parser.add_argument(
        '--gobig_prescaled',
        action='store_true',
        required=False,
        help='Add this option if you have already upscaled the image you want to gobig on. The image and its resolution will be used.'
    )
    my_parser.add_argument(
        '--device',
        action='store',
        default = "cuda:0",
        required=False,
        help='The device to use for pytorch.'
    )
    my_parser.add_argument(
        '--interactive',
        action='store_true',
        required=False,
        help='Advanced option for bots and such. Wait for a job file, render it, then wait some more.'
    )

    return my_parser.parse_args()

# Simple check to see if a key is present in the settings file
def is_json_key_present(json, key, subkey="none"):
    try:
        if subkey != "none":
            buf = json[key][subkey]
        else:
            buf = json[key]
    except KeyError:
        return False
    if type(buf) == type(None):
        return False
    return True

# pick a random item from the cooresponding text file
def randomizer(category):
    random.seed()
    randomizers = []
    with open(f'settings/{category}.txt', encoding="utf-8") as f:
        for line in f:
            randomizers.append(line.strip())
    random_item = random.choice(randomizers)
    return(random_item)

# replace anything surrounded by underscores with a random entry from the matching text file
def randomize_prompt(prompt):
    while "_" in prompt:
        start = prompt.index('_')
        end = prompt.index('_', start+1)
        swap = prompt[(start + 1):end]
        swapped = randomizer(swap)
        prompt = prompt.replace(f'_{swap}_', swapped, 1)
    return prompt

# Dynamic value - takes ready-made possible options within a string and returns the string with an option randomly selected
# Format is "I will return <Value1|Value2|Value3> in this string"
# Which would come back as "I will return Value2 in this string" (for example)
# Optionally if a value of ^^# is first, it means to return that many dynamic values,
# so <^^2|Value1|Value2|Value3> in the above example would become:
# "I will return Value3 Value2 in this string"
# note: for now assumes a string for return. TODO return a desired type
def dynamic_value(incoming):
    if type(incoming) == str:  # we only need to do something if it's a string...
        if incoming == "auto" or incoming == "random":
            return incoming
        elif "<" in incoming:   # ...and if < is in the string...
            text = incoming
            while "<" in text:
                start = text.find('<')
                end = text.find('>')
                swap = text[(start + 1):end]
                value = ""
                count = 1
                values = swap.split('|')
                if "^^" in values[0]:
                    count = values[0]
                    values.pop(0)
                    count = int(count[2:])
                random.shuffle(values)
                for i in range(count):
                    value = value + values[i] + " "
                value = value[:-1]  # remove final space
                text = text.replace(f'<{swap}>', value, 1)
            return text
        else:
            return incoming
    else:
        return incoming

class Settings:
    prompt = "A druid in his shop, selling potions and trinkets, fantasy painting by raphael lacoste and craig mullins"
    batch_name = "default"
    out_path = "./out"
    n_batches = 1
    steps = 50
    eta = 0.0
    n_iter = 1
    width = 512
    height = 512
    scale = 5.0
    dyn = None
    from_file = None
    seed = "random"
    variance = 0.0
    frozen_seed = False
    init_image = None
    init_strength = 0.5
    gobig = False
    gobig_init = None
    gobig_prescaled = False
    gobig_maximize = True
    gobig_overlap = 64
    gobig_realesrgan = False
    gobig_keep_slices = False
    esrgan_model = "realesrgan-x4plus"
    cool_down = 0.0
    checkpoint = "./models/sd-v1-4.ckpt"
    use_jpg = False
    hide_metadata = False
    method = "k_lms"
    save_settings = False
    
    def apply_settings_file(self, filename, settings_file):
        print(f'Applying settings file: {filename}')
        if is_json_key_present(settings_file, 'prompt'):
            self.prompt = (settings_file["prompt"])
        if is_json_key_present(settings_file, 'batch_name'):
            self.batch_name = (settings_file["batch_name"])
        if is_json_key_present(settings_file, 'out_path'):
            self.out_path = (settings_file["out_path"])
        if is_json_key_present(settings_file, 'n_batches'):
            self.n_batches = (settings_file["n_batches"])
        if is_json_key_present(settings_file, 'steps'):
            self.steps = (settings_file["steps"])
        if is_json_key_present(settings_file, 'eta'):
            self.eta = (settings_file["eta"])
        if is_json_key_present(settings_file, 'n_iter'):
            self.n_iter = (settings_file["n_iter"])
        if is_json_key_present(settings_file, 'width'):
            self.width = (settings_file["width"])
        if is_json_key_present(settings_file, 'height'):
            self.height = (settings_file["height"])
        if is_json_key_present(settings_file, 'scale'):
            self.scale = (settings_file["scale"])
        if is_json_key_present(settings_file, 'dyn'):
            self.dyn = (settings_file["dyn"])
        if is_json_key_present(settings_file, 'from_file'):
            self.from_file = (settings_file["from_file"])
        if is_json_key_present(settings_file, 'seed'):
            self.seed = (settings_file["seed"])
            if self.seed == "random":
                self.seed = random.randint(1, 10000000)
        if is_json_key_present(settings_file, 'variance'):
            self.variance = (settings_file["variance"])
        if is_json_key_present(settings_file, 'frozen_seed'):
            self.frozen_seed = (settings_file["frozen_seed"])
        if is_json_key_present(settings_file, 'init_strength'):
            self.init_strength = (settings_file["init_strength"])
        if is_json_key_present(settings_file, 'init_image'):
            self.init_image = (settings_file["init_image"])
        if is_json_key_present(settings_file, 'gobig'):
            self.gobig = (settings_file["gobig"])
        if is_json_key_present(settings_file, 'gobig_init'):
            self.gobig_init = (settings_file["gobig_init"])
        if is_json_key_present(settings_file, 'gobig_scale'):
            self.gobig_scale = (settings_file["gobig_scale"])
        if is_json_key_present(settings_file, 'gobig_prescaled'):
            self.gobig_prescaled = (settings_file["gobig_prescaled"])
        if is_json_key_present(settings_file, 'gobig_maximize'):
            self.gobig_maximize = (settings_file["gobig_maximize"])
        if is_json_key_present(settings_file, 'gobig_overlap'):
            self.gobig_overlap = (settings_file["gobig_overlap"])
        if is_json_key_present(settings_file, 'gobig_realesrgan'):
            self.gobig_realesrgan = (settings_file["gobig_realesrgan"])
        if is_json_key_present(settings_file, 'esrgan_model'):
            self.esrgan_model = (settings_file["esrgan_model"])
        if is_json_key_present(settings_file, 'gobig_keep_slices'):
            self.gobig_keep_slices = (settings_file["gobig_keep_slices"])
        if is_json_key_present(settings_file, 'cool_down'):
            self.cool_down = (settings_file["cool_down"])
        if is_json_key_present(settings_file, 'checkpoint'):
            self.checkpoint = (settings_file["checkpoint"])
        if is_json_key_present(settings_file, 'use_jpg'):
            self.use_jpg = (settings_file["use_jpg"])
        if is_json_key_present(settings_file, 'hide_metadata'):
            self.hide_metadata = (settings_file["hide_metadata"])
        if is_json_key_present(settings_file, 'method'):
            self.method = (settings_file["method"])
        if is_json_key_present(settings_file, 'save_settings'):
            self.save_settings = (settings_file["save_settings"])

def save_settings(options, prompt, filenum):
    setting_list = {
        'prompt' : prompt,
        'batch_name' : options.batch_name,
        'steps' : options.ddim_steps,
        'eta' : options.ddim_eta,
        'n_iter' : options.n_iter,
        'width' : options.W,
        'height' : options.H,
        'scale' : options.scale,
        'dyn' : options.dyn,
        'seed' : options.seed,
        'variance' : options.variance,
        'init_image' : options.init_image,
        'init_strength' : 1.0 - options.strength,
        'gobig' : options.gobig,
        'gobig_init' : options.gobig_init,
        'gobig_scale' : options.gobig_scale,
        'gobig_prescaled' : options.gobig_prescaled,
        'gobig_maximize' : options.gobig_maximize,
        'gobig_overlap' : options.gobig_overlap,
        'gobig_realesrgan' : options.gobig_realesrgan,
        'gobig_keep_slices' : options.gobig_keep_slices,
        'esrgan_model': options.esrgan_model,
        'use_jpg' : "true" if options.filetype == ".jpg" else "false",
        'hide_metadata' : options.hide_metadata,
        'method' : options.method
    }
    with open(f"{options.outdir}/{options.batch_name}-{filenum:04}.json",  "w+", encoding="utf-8") as f:
        json.dump(setting_list, f, ensure_ascii=False, indent=4)

def esrgan_resize(input, id, esrgan_model='realesrgan-x4plus'):
    input.save(f'_esrgan_orig{id}.png')
    input.close()
    try:
        subprocess.run(
            ['realesrgan-ncnn-vulkan', '-n', esrgan_model, '-i', '_esrgan_orig.png', '-o', '_esrgan_.png'],
            stdout=subprocess.PIPE
        ).stdout.decode('utf-8')
        output = Image.open('_esrgan_.png').convert('RGBA')
        return output
    except Exception as e:
        print('ESRGAN resize failed. Make sure realesrgan-ncnn-vulkan is in your path (or in this directory)')
        print(e)
        quit()

def do_gobig(gobig_init, device, model, opt):
    overlap = opt.gobig_overlap
    outpath = opt.outdir
    # get our render size for each slice, and our target size
    input_image = Image.open(gobig_init).convert('RGBA')
    if opt.gobig_prescaled == False:
        opt.W, opt.H = input_image.size
        target_W = opt.W * opt.gobig_scale
        target_H = opt.H * opt.gobig_scale
        if opt.gobig_realesrgan:
            input_image = esrgan_resize(input_image, opt.device_id, opt.esrgan_model)
        target_image = input_image.resize((target_W, target_H), get_resampling_mode()) #esrgan resizes 4x by default, so this brings us in line with our actual scale target
    else:
        #target_W, target_H = input_image.size
        target_image = input_image
    slices, target_image = grid_slice(target_image, overlap, (opt.W, opt.H), opt.gobig_maximize)
    # now we trigger a do_run for each slice
    betterslices = []
    slice_image = f'slice{opt.device_id}.png'
    for count, chunk_w_coords in enumerate(slices):
        chunk, coord_x, coord_y = chunk_w_coords
        chunk.save(slice_image)
        chunk.close()
        opt.init_image = slice_image
        opt.save_settings = False # we don't need to keep settings for each slice, just the main image.
        opt.n_iter = 1 # no point doing multiple iterations since only one will be used
        opt.seed = opt.seed + 1
        result = do_run(device, model, opt)
        resultslice = Image.open(result).convert('RGBA')
        betterslices.append((resultslice.copy(), coord_x, coord_y))
        resultslice.close()
        if opt.gobig_keep_slices == False:
            os.remove(result)
    # create an alpha channel for compositing the slices
    alpha = Image.new('L', (opt.W, opt.H), color=0xFF)
    alpha_gradient = ImageDraw.Draw(alpha)
    a = 0
    i = 0
    a_overlap = int(overlap / 2) # we want the alpha gradient to be half the size of the overlap, otherwise we always see some of the original background underneath
    shape = ((opt.W, opt.H), (0,0))
    while i < overlap:
        alpha_gradient.rectangle(shape, fill = a)
        a += int(255 / a_overlap)
        a = 255 if a > 255 else a
        i += 1
        shape = ((opt.W - i, opt.H - i), (i,i))
    mask = Image.new('RGBA', (opt.W, opt.H), color=0)
    mask.putalpha(alpha)
    # now composite the slices together
    finished_slices = []
    for betterslice, x, y in betterslices:
        finished_slice = addalpha(betterslice, mask)
        finished_slices.append((finished_slice, x, y))
    final_output = grid_merge(target_image, finished_slices)
    # name the file in a way that hopefully doesn't break things
    print(f'result is {result}')
    result = result.replace('.png','')
    result_split = result.rsplit('-', 1)
    result_split[0] = result_split[0] + '_gobig-'
    result = result_split[0] + result_split[1]
    print(f'Gobig output saved as {result}{opt.filetype}')
    final_output.save(f'{result}{opt.filetype}', quality = opt.quality)
    final_output.close()
    input_image.close()

def main():
    print('\nPROG ROCK STABLE')
    print('----------------')

    # rolling a d20 to see if I should pester you about supporting PRD.
    # Apologies if this offends you. At least it's only on a critical miss, right?
    d20 = random.randint(1, 20)
    if d20 == 1:
        print('Please consider supporting my Patreon. Thanks! https://is.gd/rVX6IH')
    else:
        print('')

    cl_args = parse_args()

    # Load the JSON config files
    settings = Settings()
    for setting_arg in cl_args.settings:
        try:
            with open(setting_arg, 'r', encoding="utf-8") as json_file:
                print(f'Parsing {setting_arg}')
                settings_file = json.load(json_file)
                settings.apply_settings_file(setting_arg, settings_file)
        except Exception as e:
            print('Failed to open or parse ' + setting_arg + ' - Check formatting.')
            print(e)
            quit()

    # override settings from files with anything coming in from the command line
    if cl_args.prompt:
        settings.prompt = cl_args.prompt

    if cl_args.output:
        settings.batch_name = cl_args.output

    if cl_args.n_batches:
        settings.n_batches = cl_args.n_batches

    if cl_args.n_iter:
        settings.n_iter = cl_args.n_iter

    if cl_args.from_file:
        settings.from_file = cl_args.from_file

    if cl_args.seed:
        settings.seed = cl_args.seed

    if cl_args.gobig:
        settings.gobig = cl_args.gobig

    if cl_args.gobig_init:
        settings.gobig_init = cl_args.gobig_init

    if cl_args.gobig_prescaled:
        settings.gobig_prescaled = cl_args.gobig_prescaled

    valid_methods = ['k_lms', 'k_dpm_2_ancestral', 'k_dpm_2', 'k_heun', 'k_euler_ancestral', 'k_euler', 'ddim']
    if any(settings.method in s for s in valid_methods):
        print(f'Using {settings.method} sampling method.')
    else:
        print(f'Method {settings.method} is not available. The valid choices are:')
        print(valid_methods)
        print()
        print(f'Falling back k_lms')
        settings.method = 'k_lms'

    # setup the model
    ckpt = settings.checkpoint # "./models/sd-v1-3-full-ema.ckpt"
    inf_config = "./configs/stable-diffusion/v1-inference.yaml"
    print(f'Loading the model and checkpoint ({ckpt})...')
    config = OmegaConf.load(f"{inf_config}")
    model = load_model_from_config(config, f"{ckpt}", verbose=False)

    # setup the device
    device_id = "" # leave this blank unless it's a cuda device
    if torch.cuda.is_available() and "cuda" in cl_args.device:
        device = torch.device(f'{cl_args.device}')
        device_id = ("_" + cl_args.device.rsplit(':',1)[1]) if "0" not in cl_args.device else ""
    elif ("mps" in cl_args.device) or (torch.backends.mps.is_available()):
        device = torch.device("mps")
        settings.method = "ddim" # k_diffusion currently not working on anything other than cuda
    else:
        # fallback to CPU if we don't recognize the device name given
        device = torch.device("cpu")
        cores = os.cpu_count()
        torch.set_num_threads(cores)
        settings.method = "ddim" # k_diffusion currently not working on anything other than cuda

    print('Pytorch is using device:', device)

    if "cuda" in str(device):
        model.cuda()
    model.eval()

    # load the model to the device
    if "cuda" in str(device):
        model = model.half() # half-precision mode for gpus, saves vram, good good
    model = model.to(device)

    there_is_work_to_do = True
    while there_is_work_to_do:
        if cl_args.interactive:
            # Interactive mode waits for a job json, runs it, then goes back to waiting
            job_json = ("job_" + cl_args.device + ".json").replace(":","_")
            print(f'\nInteractive Mode On! Waiting for {job_json}')
            job_ready = False
            while job_ready == False:
                if os.path.exists(job_json):
                    print(f'Job file found! Processing.')
                    try:
                        with open(job_json, 'r', encoding="utf-8") as json_file:
                            settings_file = json.load(json_file)
                            settings.apply_settings_file(job_json, settings_file)
                            prompts = []
                            prompts.append(settings.prompt)
                        job_ready = True
                    except Exception as e:
                        print('Failed to open or parse ' + job_json + ' - Check formatting.')
                        print(e)
                        os.remove(job_json)
                else:
                    time.sleep(0.5)

        #outdir = (f'{settings.out_path}/{settings.batch_name}')
        outdir = os.path.join(settings.out_path, settings.batch_name)
        print(f'Saving output to {outdir}')
        filetype = ".jpg" if settings.use_jpg == True else ".png"
        quality = 97 if settings.use_jpg else 100

        prompts = []
        if settings.from_file is not None:
            with open(settings.from_file, "r", encoding="utf-8") as f:
                prompts = f.read().splitlines()
        else:
            prompts.append(settings.prompt)


        for p in range(len(prompts)):
            for i in range(settings.n_batches):
                # pack up our settings into a simple namespace for the renderer
                opt = {
                    "prompt" : prompts[p],
                    "batch_name" : settings.batch_name,
                    "outdir" : outdir,
                    "ddim_steps" : settings.steps,
                    "ddim_eta" : settings.eta,
                    "n_iter" : settings.n_iter,
                    "W" : settings.width,
                    "H" : settings.height,
                    "C" : 4,
                    "f" : 8,
                    "scale" : settings.scale,
                    "dyn" : settings.dyn,
                    "seed" : settings.seed + i,
                    "variance": settings.variance,
                    "variance_seed": settings.seed + i + 1,
                    "precision": "autocast",
                    "init_image": settings.init_image,
                    "strength": 1.0 - settings.init_strength,
                    "gobig": settings.gobig,
                    "gobig_init": settings.gobig_init,
                    "gobig_scale": settings.gobig_scale,
                    "gobig_prescaled": settings.gobig_prescaled,
                    "gobig_maximize": settings.gobig_maximize,
                    "gobig_overlap": settings.gobig_overlap,
                    "gobig_realesrgan": settings.gobig_realesrgan,
                    "gobig_keep_slices": settings.gobig_keep_slices,
                    "esrgan_model": settings.esrgan_model,
                    "config": config,
                    "filetype": filetype,
                    "hide_metadata": settings.hide_metadata,
                    "quality": quality,
                    "device_id": device_id,
                    "method": settings.method,
                    "save_settings": settings.save_settings,
                }
                opt = SimpleNamespace(**opt)

                # render the image(s)!
                if settings.gobig_init == None:
                    # either just a regular render, or a regular render that will next go_big
                    gobig_init = do_run(device, model, opt)
                else:
                    gobig_init = settings.gobig_init
                if settings.gobig:
                    do_gobig(gobig_init, device, model, opt)
                if settings.cool_down > 0 and ((i < (settings.n_batches - 1)) or p < (len(prompts) - 1)):
                    print(f'Pausing {settings.cool_down} seconds to give your poor GPU a rest...')
                    time.sleep(settings.cool_down)
            if not settings.frozen_seed:
                settings.seed = settings.seed + 1
        if cl_args.interactive == False:
            #only doing one render, so we stop after this
            there_is_work_to_do = False
        else:
            print('\nJob finished! And so we wait...\n')
            os.remove(job_json)

if __name__ == "__main__":
    main()

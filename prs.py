import argparse, os, sys, glob
import random
import torch
import numpy as np
from omegaconf import OmegaConf
import PIL
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from einops import rearrange, repeat
from tqdm import tqdm, trange
from itertools import islice
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
#import accelerate
from contextlib import contextmanager, nullcontext

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

import json5 as json

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

def do_run(device, model, opt):
    print('Starting render!')
    from types import SimpleNamespace
    opt = SimpleNamespace(**opt)
    seed_everything(opt.seed)

    if opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    batch_size = opt.n_samples
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
    if not opt.from_file:
        prompt = opt.prompt
        assert prompt is not None
        data = [batch_size * [prompt]]

    else:
        #TODO: process a prompt file ahead of time for randomizers
        print(f"reading prompts from {opt.from_file}")
        with open(opt.from_file, "r") as f:
            data = f.read().splitlines()
            data = list(chunk(data, batch_size))

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1

    start_code = None
    if opt.fixed_code:
        start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)

    precision_scope = autocast if opt.precision=="autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                tic = time.time()
                all_samples = list()
                for n in trange(opt.n_iter, desc="Sampling"):
                    for prompts in tqdm(data, desc="data"):
                        uc = None
                        if opt.scale != 1.0:
                            uc = model.get_learned_conditioning(batch_size * [""])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        c = model.get_learned_conditioning(prompts)
                        shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                        samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                         conditioning=c,
                                                         batch_size=opt.n_samples,
                                                         shape=shape,
                                                         verbose=False,
                                                         unconditional_guidance_scale=opt.scale,
                                                         unconditional_conditioning=uc,
                                                         eta=opt.ddim_eta,
                                                         x_T=start_code)

                        x_samples_ddim = model.decode_first_stage(samples_ddim)
                        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

                        if not opt.skip_save:
                            for x_sample in x_samples_ddim:
                                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                Image.fromarray(x_sample.astype(np.uint8)).save(
                                    os.path.join(sample_path, f"{base_count:05}.png"))
                                base_count += 1

                        if not opt.skip_grid:
                            all_samples.append(x_samples_ddim)

                if not opt.skip_grid:
                    # additionally, save as grid
                    grid = torch.stack(all_samples, 0)
                    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                    grid = make_grid(grid, nrow=n_rows)

                    add_metadata = True
                    metadata = PngInfo()
                    if add_metadata == True:
                        metadata.add_text("prompt", opt.prompt)
                        metadata.add_text("seed", str(opt.seed))
                        metadata.add_text("steps", str(opt.ddim_steps))

                    # to image
                    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                    Image.fromarray(grid.astype(np.uint8)).save(os.path.join(outpath, f'{opt.batch_name}-{grid_count:04}.png'), pnginfo=metadata)
                    grid_count += 1

                toc = time.time()

def parse_args():
    my_parser = argparse.ArgumentParser(
        prog='MathRockDiffusion',
        description='Generate images from text prompts.',
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
        help='How many images to generate'
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
                start = text.index('<')
                end = text.index('>')
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
                text = text.replace(f'<{swap}>', value)
            return text
        else:
            return incoming
    else:
        return incoming

class Settings:
    prompt = "A druid in his shop, selling potions and trinkets, fantasy painting by raphael lacoste and craig mullins"
    batch_name = "default"
    n_batches = 1
    skip_grid = False
    skip_save = False
    steps = 50
    plms = False
    eta = 0.0
    n_iter = 1
    width = 512
    height = 512
    n_samples = 1
    n_rows = 2
    scale = 5.0
    dyn = None
    from_file = None
    seed = "random"
    
    def apply_settings_file(self, filename, settings_file):
        print(f'Applying settings file: {filename}')
        if is_json_key_present(settings_file, 'prompt'):
            self.prompt = (settings_file["prompt"])
        if is_json_key_present(settings_file, 'batch_name'):
            self.batch_name = (settings_file["batch_name"])
        if is_json_key_present(settings_file, 'n_batches'):
            self.n_batches = (settings_file["n_batches"])
        if is_json_key_present(settings_file, 'skip_grid'):
            self.skip_grid = (settings_file["skip_grid"])
        if is_json_key_present(settings_file, 'skip_save'):
            self.skip_save = (settings_file["skip_save"])
        if is_json_key_present(settings_file, 'steps'):
            self.steps = (settings_file["steps"])
        if is_json_key_present(settings_file, 'plms'):
            self.plms = (settings_file["plms"])
        if is_json_key_present(settings_file, 'eta'):
            self.eta = (settings_file["eta"])
        if is_json_key_present(settings_file, 'n_iter'):
            self.n_iter = (settings_file["n_iter"])
        if is_json_key_present(settings_file, 'width'):
            self.width = (settings_file["width"])
        if is_json_key_present(settings_file, 'height'):
            self.height = (settings_file["height"])
        if is_json_key_present(settings_file, 'n_samples'):
            self.n_samples = (settings_file["n_samples"])
        if is_json_key_present(settings_file, 'n_rows'):
            self.n_rows = (settings_file["n_rows"])
        if is_json_key_present(settings_file, 'scale'):
            self.scale = (settings_file["scale"])
        if is_json_key_present(settings_file, 'dyn'):
            self.dyn = (settings_file["dyn"])
        if is_json_key_present(settings_file, 'from_file'):
            self.from_file = (settings_file["from_file"])
        if is_json_key_present(settings_file, 'dyn'):
            self.dyn = (settings_file["dyn"])
        if is_json_key_present(settings_file, 'seed'):
            self.seed = (settings_file["seed"])
            if self.seed == "random":
                self.seed = random.randint(1, 10000000)

def main():
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

    outdir = (f'./out/{settings.batch_name}')

    # process the prompt for randomizers and dynamic values
    settings.prompt = randomize_prompt(settings.prompt)
    settings.prompt = dynamic_value(settings.prompt)
    print(f'Setting prompt to: \n"{settings.prompt}"\n')

    #accelerator = accelerate.Accelerator()
    ckpt = "./models/sd-v1-3-full-ema.ckpt"
    inf_config = "./configs/stable-diffusion/v1-inference.yaml"
    config = OmegaConf.load(f"{inf_config}")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = load_model_from_config(config, f"{ckpt}", verbose=False)
    model = model.to(device)

    for i in range(settings.n_batches):
        # pack up our settings into a simple namespace for the renderer
        opt = {
            "prompt" : settings.prompt,
            "batch_name" : settings.batch_name,
            "outdir" : outdir,
            "skip_grid" : settings.skip_grid,
            "skip_save" : settings.skip_save,
            "ddim_steps" : settings.steps,
            "plms" : settings.plms,
            "ddim_eta" : settings.eta,
            "n_iter" : settings.n_iter,
            "W" : settings.width,
            "H" : settings.height,
            "C" : 4,
            "f" : 8,
            "n_samples" : settings.n_samples,
            "n_rows" : settings.n_rows,
            "scale" : settings.scale,
            "dyn" : settings.dyn,
            "from_file": settings.from_file,
            "seed" : settings.seed,
            "fixed_code": False,
            "precision": "autocast",
            "config": config
        }
        # render the image(s)!
        do_run(device, model, opt)

if __name__ == "__main__":
    main()

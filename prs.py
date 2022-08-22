import argparse, os, sys, glob
import random
import torch
import numpy as np
from omegaconf import OmegaConf
import PIL
from PIL import Image, ImageOps, ImageStat, ImageEnhance, ImageDraw
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
import subprocess

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

from types import SimpleNamespace
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
    print(f"loaded input image of size ({w}, {h}) from {path}")
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), get_resampling_mode())
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.


def do_run(device, model, opt):
    print('Starting render!')
    seed_everything(opt.seed)

    if opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    batch_size = opt.n_samples
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size

    # prompt = opt.prompt
    data = [batch_size * [opt.prompt]]
    # data = opt.prompt

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1

    if opt.init_image is not None:
        assert os.path.isfile(opt.init_image)
        init_image = load_img(opt.init_image).to(device)
        init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
        init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))  # move to latent space
        sampler.make_schedule(ddim_num_steps=opt.ddim_steps, ddim_eta=opt.ddim_eta, verbose=False)

        # strength is like skip_steps I think
        assert 0. <= opt.strength <= 1., 'can only work with strength in [0.0, 1.0]'
        t_enc = int(opt.strength * opt.ddim_steps)
        print(f"target t_enc is {t_enc} steps")
    else:
        init_image = None

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

                        # process the prompt for randomizers and dynamic values
                        newprompts = []
                        for prompt in prompts:
                            prompt = randomize_prompt(prompt)
                            prompt = dynamic_value(prompt)
                            newprompts.append(prompt)
                        prompts = newprompts

                        print(f'\nPrompt for this image:\n   {prompts}\n')

                        c = model.get_learned_conditioning(prompts)

                        if init_image is None:
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

                        else:
                            # encode (scaled latent)
                            z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc]*batch_size).to(device))
                            # decode it
                            samples_ddim = sampler.decode(z_enc, c, t_enc, unconditional_guidance_scale=opt.scale,
                                                    unconditional_conditioning=uc,)


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
                        metadata.add_text("prompt", str(prompts))
                        metadata.add_text("seed", str(opt.seed))
                        metadata.add_text("steps", str(opt.ddim_steps))

                    # to image
                    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                    output_filename = os.path.join(outpath, f'{opt.batch_name}-{grid_count:04}.png')
                    Image.fromarray(grid.astype(np.uint8)).save(output_filename, pnginfo=metadata)
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

def grid_coords(target, original, overlap):
    #generate a list of coordinate tuples for our sections, in order of how they'll be rendered
    #target should be the size for the gobig result, original is the size of each chunk being rendered
    center = []
    target_x, target_y = target
    center_x = int(target_x / 2)
    center_y = int(target_y / 2)
    original_x, original_y = original
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
    # calculate a new size that will fill the canvas, which will be optionally used in grid_slice and go_big
    last_coordx, last_coordy = dy_list[-1:][0]
    render_edgey = last_coordy + original_y # outer bottom edge of the render canvas
    render_edgex = last_coordx + original_x # outer side edge of the render canvas
    scalarx = render_edgex / target_x
    scalary = render_edgey / target_y
    if scalarx <= scalary:
        new_edgex = int(target_x * scalarx)
        new_edgey = int(target_y * scalarx)
    else:
        new_edgex = int(target_x * scalary)
        new_edgey = int(target_y * scalary)
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
    return result, (new_edgex, new_edgey)

# Chop our source into a grid of images that each equal the size of the original render
def grid_slice(source, overlap, og_size): 
    width, height = og_size # size of the slices to be rendered
    coordinates, new_size = grid_coords(source.size, og_size, overlap)
    slices = []
    for coordinate in coordinates:
        x, y = coordinate
        slices.append(((source.crop((x, y, x+width, y+height))), x, y))
    global slices_todo
    slices_todo = len(slices) - 1
    return slices, new_size

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
        help='How many images to generate'
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
    init_image = None
    init_strength = 0.5
    gobig_maximize = True
    gobig_overlap = 64
    gobig_realesrgan = False
    cool_down = 0.0
    checkpoint = "./models/sd-v1-4.ckpt"
    
    def apply_settings_file(self, filename, settings_file):
        print(f'Applying settings file: {filename}')
        if is_json_key_present(settings_file, 'prompt'):
            self.prompt = (settings_file["prompt"])
        if is_json_key_present(settings_file, 'batch_name'):
            self.batch_name = (settings_file["batch_name"])
        if is_json_key_present(settings_file, 'n_batches'):
            self.n_batches = (settings_file["n_batches"])
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
        if is_json_key_present(settings_file, 'seed'):
            self.seed = (settings_file["seed"])
            if self.seed == "random":
                self.seed = random.randint(1, 10000000)
        if is_json_key_present(settings_file, 'init_strength'):
            self.init_strength = (settings_file["init_strength"])
        if is_json_key_present(settings_file, 'init_image'):
            self.init_image = (settings_file["init_image"])
        if is_json_key_present(settings_file, 'gobig_maximize'):
            self.gobig_maximize = (settings_file["gobig_maximize"])
        if is_json_key_present(settings_file, 'gobig_overlap'):
            self.gobig_overlap = (settings_file["gobig_overlap"])
        if is_json_key_present(settings_file, 'gobig_realesrgan'):
            self.gobig_realesrgan = (settings_file["gobig_realesrgan"])
        if is_json_key_present(settings_file, 'cool_down'):
            self.cool_down = (settings_file["cool_down"])
        if is_json_key_present(settings_file, 'checkpoint'):
            self.checkpoint = (settings_file["checkpoint"])

def esrgan_resize(input):
    input.save('_esrgan_orig.png')
    try:
        subprocess.run(
            ['realesrgan-ncnn-vulkan', '-i', '_esrgan_orig.png', '-o', '_esrgan_.png'],
            stdout=subprocess.PIPE
        ).stdout.decode('utf-8')
        output = Image.open('_esrgan_.png').convert('RGBA')
        return output
    except Exception as e:
        print('ESRGAN resize failed. Make sure realesrgan-ncnn-vulkan is in your path (or in this directory)')
        print(e)
        quit()

def do_gobig(gobig_init, gobig_scale, device, model, opt):
    overlap = opt.gobig_overlap
    outpath = opt.outdir
    # get our render size for each slice, and our target size
    input_image = Image.open(gobig_init).convert('RGBA')
    opt.W, opt.H = input_image.size
    target_W = opt.W * gobig_scale
    target_H = opt.H * gobig_scale
    if opt.gobig_realesrgan:
        input_image = esrgan_resize(input_image)
    target_image = input_image.resize((target_W, target_H), get_resampling_mode())
    slices, new_canvas_size = grid_slice(target_image, overlap, (opt.W, opt.H))
    if opt.gobig_maximize == True:
        # increase our final image size to use up blank space
        target_image = input_image.resize(new_canvas_size, get_resampling_mode())
        slices, new_canvas_size = grid_slice(target_image, overlap, (opt.W, opt.H))
    input_image.close()
    # now we trigger a do_run for each slice
    betterslices = []
    slice_image = 'slice.png'
    opt.seed = opt.seed + 1
    for count, chunk_w_coords in enumerate(slices):
        chunk, coord_x, coord_y = chunk_w_coords
        chunk.save(slice_image)
        opt.init_image = slice_image
        result = do_run(device, model, opt)
        resultslice = Image.open(result).convert('RGBA')
        betterslices.append((resultslice.copy(), coord_x, coord_y))
        resultslice.close()
    # create an alpha channel for compositing the slices
    alpha = Image.new('L', (opt.W, opt.H), color=0xFF)
    alpha_gradient = ImageDraw.Draw(alpha)
    a = 0
    i = 0
    shape = ((opt.W, opt.H), (0,0))
    while i < overlap:
        alpha_gradient.rectangle(shape, fill = a)
        a += int(255 / overlap)
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
    final_output.save(f'{result}-gobig.png')

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

    if cl_args.from_file:
        settings.from_file = cl_args.from_file

    outdir = (f'./out/{settings.batch_name}')

    # setup the model
    ckpt = settings.checkpoint # "./models/sd-v1-3-full-ema.ckpt"
    inf_config = "./configs/stable-diffusion/v1-inference.yaml"
    print('Loading the model and checkpoint...')
    config = OmegaConf.load(f"{inf_config}")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = load_model_from_config(config, f"{ckpt}", verbose=False)
    model = model.to(device)

    prompts = []
    if settings.from_file is not None:
        with open(settings.from_file, "r") as f:
            prompts = f.read().splitlines()
    else:
        prompts.append(settings.prompt)

    for prompt in prompts:
        for i in range(settings.n_batches):
            # pack up our settings into a simple namespace for the renderer
            opt = {
                "prompt" : prompt,
                "batch_name" : settings.batch_name,
                "outdir" : outdir,
                "skip_grid" : False,
                "skip_save" : False,
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
                "seed" : settings.seed + i,
                "fixed_code": False,
                "precision": "autocast",
                "init_image": settings.init_image,
                "strength": 1.0 - settings.init_strength,
                "gobig_maximize": settings.gobig_maximize,
                "gobig_overlap": settings.gobig_overlap,
                "gobig_realesrgan": settings.gobig_realesrgan,
                "config": config
            }
            opt = SimpleNamespace(**opt)
            print(f'Opt steps is {opt.ddim_steps}')
            # render the image(s)!
            if cl_args.gobig_init == None:
                # either just a regular render, or a regular render that will next go_big
                gobig_init = do_run(device, model, opt)
            else:
                gobig_init = cl_args.gobig_init
            if cl_args.gobig:
                do_gobig(gobig_init, cl_args.gobig_scale, device, model, opt)
            if settings.cool_down > 0 and i < (settings.n_batches - 1):
                print(f'Pausing {settings.cool_down} seconds to give your poor GPU a rest...')
                time.sleep(settings.cool_down)

if __name__ == "__main__":
    main()

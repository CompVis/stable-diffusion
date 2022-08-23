import PIL
import gradio as gr
import argparse, os, sys, glob
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
import accelerate
import random
import pynvml
import threading

import k_diffusion as K
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

parser = argparse.ArgumentParser()
parser.add_argument(
    "--txt2img_outdir",
    type=str,
    nargs="?",
    default="outputs/txt2img-samples",
    help="dir to write txt2img results to",
)
parser.add_argument(
    "--img2img_outdir",
    type=str,
    nargs="?",
    default="outputs/img2img-samples",
    help="dir to write img2img results to",
)
parser.add_argument(
    "--txt2img_url",
    type=str,
    nargs="?",
    help="if you've also got a webserver that you can output to, you can put the URL of the appropriate output folder here for grid link auto-generation",
)
parser.add_argument(
    "--img2img_url",
    type=str,
    nargs="?",
)
opt = parser.parse_args()

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

def torch_gc():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

def seed_to_int(s):
    if s == '':
        return random.randint(0,2**32)
    n = abs(int(s) if s.isdigit() else hash(s))
    while n > 2**32:
        n = n >> 32
    return n

def crash(e, s):
    global model
    global device

    del model
    del device

    print(s, '\n', e)
    
    print('exiting...calling os._exit(0)')
    
    # force-kill the process, because it always freezes for me otherwise
    t = threading.Timer(0.25, os._exit, args=[0])
    t.start()

class MemUsageMonitor(threading.Thread):
    stop_flag = False
    max_usage = 0
    total = 0
    
    def __init__(self, name):
        threading.Thread.__init__(self)
        self.name = name
    
    def run(self):
        print(f"[{self.name}] Recording max memory usage...\n")
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.total = pynvml.nvmlDeviceGetMemoryInfo(handle).total
        while not self.stop_flag:
            m = pynvml.nvmlDeviceGetMemoryInfo(handle)
            self.max_usage = max(self.max_usage, m.used)
            # print(self.max_usage)
            time.sleep(0.1)
        print(f"[{self.name}] Stopped recording.\n")
        pynvml.nvmlShutdown()
    
    def read(self):
        return self.max_usage, self.total
    
    def stop(self):
        self.stop_flag = True
    
    def read_and_stop(self):
        self.stop_flag = True
        return self.max_usage, self.total

config = OmegaConf.load("configs/stable-diffusion/v1-inference.yaml")
model = load_model_from_config(config, "models/ldm/stable-diffusion-v1/model.ckpt")

device = torch.device("cuda")
model = model.half().to(device)

def unified(prompt: str, init_img, ddim_steps: int, sampler_index: int, toggles: list, ddim_eta: float, n_iter: int, n_samples: int, cfg_scale: float, denoising_strength: float, seed: str, width: int, height: int, channels: int, _):
    err = False
    start_time = time.time()
    torch_gc()
    
    downsampling_factor = 8

    img2img_mode = init_img != None

    mode_str = 'Image Translation' if img2img_mode else 'Dream'

    mem_mon = MemUsageMonitor('MemMon')
    mem_mon.start()
    
    sampler_str = ['DDIM','PLMS','k'][sampler_index]

    skip_save = 0 in toggles
    skip_grid = 1 in toggles
    inc_seed = 2 in toggles

    rng_seed = seed_everything(seed_to_int(seed))

    config = "configs/stable-diffusion/v1-inference.yaml"
    ckpt = "models/ldm/stable-diffusion-v1/model.ckpt",
    outdir = opt.img2img_outdir if img2img_mode else opt.txt2img_outdir
    
    accelerator = accelerate.Accelerator()
    
    if sampler_index == 1:
        sampler = PLMSSampler(model)
        ddim_eta = 0.0;
    elif sampler_index == 2:
        sampler = None
        model_wrap = K.external.CompVisDenoiser(model)
        sigma_min, sigma_max = model_wrap.sigmas[0].item(), model_wrap.sigmas[-1].item()
    else:
        sampler = DDIMSampler(model)

    os.makedirs(outdir, exist_ok=True)
    outpath = outdir

    batch_size = n_samples
    n_rows = batch_size

    assert prompt is not None
    data = [batch_size * [prompt]]

    # part of the file name for sample saving; remove Win-incompatible characters, replace spaces with _
    prompt_filename = ''.join([c for c in prompt if c not in '\/:*?"<>|']).replace(' ', '_')

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1
    grid_file = ''

    start_code = None

    seed_inc = 0

    #precision_scope = autocast if opt.precision=="autocast" else nullcontext
    precision_scope = autocast
    output_images = []

    if img2img_mode:
        image = init_img.convert("RGB")
        w, h = image.size
        print(f"loaded input image of size ({w}, {h})")
        if width > 0:
            w = width
        if height > 0:
            h = height
        w, h = map(lambda x: x - x % 64, (w, h))  # resize to integer multiple of 64
        image = image.resize((w, h), resample=PIL.Image.LANCZOS if hasattr(PIL.Image, 'Resampling') else PIL.Image.LANCZOS)
        print(f"cropped image to size ({w}, {h})")
        image = np.array(image).astype(np.float32) / 255.0
        image = image[None].transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)
    else:
        w, h = width, height

    try:
        with torch.no_grad(), precision_scope("cuda"):
            if img2img_mode:
                init_image = 2.*image - 1.
                init_image = init_image.to(device)
                init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
                init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))  # move to latent space
                if sampler_index == 2:
                    x0 = init_latent
                else:
                    sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=ddim_eta, verbose=False)
                assert 0. <= denoising_strength <= 1., 'can only work with strength in [0.0, 1.0]'
                t_enc = int(denoising_strength * ddim_steps)
                print(f"target t_enc is {t_enc} steps")

            with model.ema_scope():
                all_samples = list()
                for n in trange(n_iter, desc="Sampling", disable=(not accelerator.is_main_process)):
                    for prompts in tqdm(data, desc="data", disable=(not accelerator.is_main_process)):
                        uc = None
                        if cfg_scale != 1.0:
                            uc = model.get_learned_conditioning(batch_size * [""])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        c = model.get_learned_conditioning(prompts)

                        if inc_seed:
                            seed_everything(rng_seed + seed_inc)

                        shape = [channels, height // downsampling_factor, width // downsampling_factor]

                        if sampler_index == 2:
                            sigmas = model_wrap.get_sigmas(ddim_steps)

                            if img2img_mode:
                                noise = torch.randn_like(x0) * sigmas[ddim_steps - t_enc - 1]
                                x = x0 + noise
                                sigmas = sigmas[ddim_steps - t_enc - 1:]
                            else:  # txt2img
                                x = torch.randn([n_samples, *shape], device=device) * sigmas[0]

                            model_wrap_cfg = CFGDenoiser(model_wrap)
                            extra_args = {'cond': c, 'uncond': uc, 'cond_scale': cfg_scale}
                            samples_ddim = K.sampling.sample_lms(model_wrap_cfg, x, sigmas, extra_args=extra_args, disable=(not accelerator.is_main_process))
                        else:
                            if img2img_mode:
                                # encode (scaled latent)
                                z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc]*batch_size).to(device))
                                # decode it
                                samples_ddim = sampler.decode(z_enc, c, t_enc,
                                                              unconditional_guidance_scale=cfg_scale,
                                                              unconditional_conditioning=uc,)
                            else:  # txt2img
                                samples_ddim, _ = sampler.sample(S=ddim_steps,
                                                                 conditioning=c,
                                                                 batch_size=n_samples,
                                                                 shape=shape,
                                                                 verbose=False,
                                                                 unconditional_guidance_scale=cfg_scale,
                                                                 unconditional_conditioning=uc,
                                                                 eta=ddim_eta,
                                                                 x_T=start_code)

                        x_samples_ddim = model.decode_first_stage(samples_ddim)
                        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                        if sampler_index == 2:
                            x_samples_ddim = accelerator.gather(x_samples_ddim)

                        if accelerator.is_main_process:
                            for n, x_sample in enumerate(x_samples_ddim):
                                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                if not skip_save:
                                    Image.fromarray(x_sample.astype(np.uint8)).save(os.path.join(sample_path, f"{base_count:05}-{sampler_str}-{rng_seed+seed_inc}s{n+1}-{prompt_filename[:128]}.png"))
                                output_images.append(Image.fromarray(x_sample.astype(np.uint8)))
                                base_count += 1

                        if inc_seed:
                            seed_inc += 1

                        if accelerator.is_main_process and not skip_grid:
                            all_samples.append(x_samples_ddim)

            if accelerator.is_main_process and not skip_grid:
                # additionally, save as grid
                grid = torch.stack(all_samples, 0)
                grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                grid = make_grid(grid, nrow=n_rows)

                # to image
                grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                grid_file = f'grid-{grid_count:04}-{sampler_str}-{rng_seed}-{prompt.replace(' ', '_')[:128]}.jpg'
                Image.fromarray(grid.astype(np.uint8)).save(os.path.join(outpath, grid_file), 'jpeg', quality=80, optimize=True)
                grid_count += 1

            toc = time.time()

        torch_gc()
        del sampler

        mem_max_used, mem_total = mem_mon.read_and_stop()
        time_diff = time.time()-start_time

        grid_link_text = ''
        if not skip_grid:
            if img2img_mode and opt.img2img_url:
                grid_link_text = f'Grid link: [<a target="_blank" href="{opt.img2img_url + grid_file}">{grid_file}</a>]'
            elif opt.txt2img_url: # txt2img mode
                grid_link_text = f'Grid link: [<a target="_blank" href="{opt.txt2img_url + grid_file}">{grid_file}</a>]'
        # reeee HTML
        out_notes = f'''<p>{prompt}</p>
<hr style="margin: 5px;">
<p>
{ddim_steps} steps
, {sampler_str} sampler
, {n_iter} iterations
, {n_samples} samples
{f', ddim ETA {ddim_eta}' if ddim_eta > 0.0 else ''}
, {cfg_scale} scale
{f', {denoising_strength} strength' if img2img_mode else ''}<br>
{w}×{h}px; seed: {rng_seed}<br>
<br>
Took { round(time_diff, 2) }s total ({ round(time_diff/(n_samples*n_iter),2) }s per image)<br>
Peak memory usage: { -(mem_max_used // -1_048_576) } MiB / { -(mem_total // -1_048_576) } MiB / { round(mem_max_used/mem_total*100, 3) }%<br>
{grid_link_text}
</p>'''
        return output_images, out_notes
    except RuntimeError as e:
        err = e
        return [], f'CRASHED:<br><textarea rows="5" style="background: black;width: -webkit-fill-available;font-family: monospace;font-size: small;font-weight: bold;">{str(e)}</textarea><br><br>Please wait while the program restarts.'
    finally:
        mem_mon.stop()
        del mem_mon

        if err:
            crash(err, f'Runtime error ({mode_str})!')

# prompt, init_img, ddim_steps, sampler_index, toggles, ddim_eta, n_iter, n_samples, cfg_scale, denoising_strength, seed, width, height, channels, _

notes_dream = '''
<script></script>

<p>Quick guide:</p>
<div>
<ul style="list-style: square; margin-left: 20px;">
  <li>'Prompt' is by far the most important setting, promptsmithing is a art. Experiment.</li>
  <li>'Sampling Steps' controls how long the model 'works on' the image, processing time increases linearly with step count. With PLMS, should be at least 20 for remotely acceptable output.
  <li>'Sampling Iterations' is for automating repeat runs with the same settings but a slightly different seed.</li>
  <li>'Samples Per Iteration' controls the number of images produced <i>per</i> iteration—slightly improves generation time per image, greatly increases VRAM usage, best around 2-4.</li>
  <li>Seed, Width, Height should be self-explanatory.</li>
</ul>
<p>Other notes:</p>
<ul style="list-style: square; margin-left: 20px;">
  <li>The model was trained on 512×512 images, so other resolutions behave unpredictably.</li>
  <li>The max resolution my VRAM can handle is 1280×640—'Samples Per Iteration' must be set to 1.</li>
  <li style="display:none;">Try 512×512, 20 steps, PLMS, 1 iteration, 6 samples for a lot of quick output at moderate quality.</li>
  <li>If you push res or sample count too high, it'll crash. Wait ~10s for auto-restart.</li>
</ul>
</div>
'''
dream_interface = gr.Interface(
    unified,
    inputs=[
        gr.Textbox(label='Prompt', placeholder="A corgi wearing a top hat as an oil painting.", lines=1),
        gr.Image(value=None, source="upload", interactive=True, type="pil", visible=False),
        gr.Slider(minimum=1, maximum=250, step=1, label="Sampling Steps", value=50),
        gr.Radio(label='Sampler', choices=["DDIM", "PLMS", "k-diffusion"], value="k-diffusion", type="index"),
        gr.CheckboxGroup(label='Toggles', choices=['Skip sample save', 'Skip grid save', 'Increment seed'], value=['Skip sample save', 'Increment seed'], type="index"),
        gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="DDIM ETA (only relevant if DDIM selected)", value=0.0, visible=True),
        gr.Slider(minimum=1, maximum=50, step=1, label='Sampling Iterations', value=2),
        gr.Slider(minimum=1, maximum=16, step=1, label='Samples Per Iteration', value=2),
        gr.Slider(minimum=1.0, maximum=20.0, step=0.1, label='Classifier-free Guidance Scale', value=7.5),
        gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label='Denoising Strength', value=0.75, visible=False),
        gr.Textbox(label="Seed (blank to randomize)", lines=1, value=""),
        gr.Slider(minimum=64, maximum=2048, step=64, label="Width", value=512),
        gr.Slider(minimum=64, maximum=2048, step=64, label="Height", value=512),
        gr.Slider(minimum=0, maximum=20, step=1, label="Latent Channels", value=4, visible=False),
        gr.HTML(notes_dream, visible=False),
    ],
    outputs=[
        gr.Gallery(),
        gr.HTML(label='Notes'),
    ],
    title="Stable Diffusion Text-to-Image K2",
    description="Generate images from text with Stable Diffusion",
    allow_flagging='never',
)

# prompt, init_img, ddim_steps, sampler_index, toggles, ddim_eta, n_iter, n_samples, cfg_scale, denoising_strength, seed, width, height, channels, _
notes_translation = '''
<p>Notes:</p>
<div>
<ul style="list-style: square; margin-left: 20px;">
  <li>See notes from Dream tab, same rules generally apply. Don't go over about 1280×640 (1 sample) or it'll run out of memory and crash.</li>
  <li>Input images should be resized and/or cropped so width <b>and</b> height are divisible by 64, otherwise the model will probably crash.</li>
  <li>If set, input image will be resized according to sliders. This will skew the image if aspect ratio mismatches.</li>
  <li>Leaving resize sliders on 0 will resize each axis to the nearest multiple of 64, rounded down.</li>
</ul>
</div>
'''
img2img_interface = gr.Interface(
    unified,
    inputs=[
        gr.Textbox(label='Prompt', placeholder="A fantasy landscape, trending on artstation.", lines=1),
        gr.Image(value="https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg", source="upload", interactive=True, type="pil"),
        gr.Slider(minimum=1, maximum=250, step=1, label="Sampling Steps", value=50),
        gr.Radio(label='Sampler', choices=["DDIM", "PLMS", "k-diffusion"], value="k-diffusion", type="index"),
        gr.CheckboxGroup(label='Toggles', choices=['Skip sample save', 'Skip grid save', 'Increment seed'], value=['Skip sample save', 'Increment seed'], type="index"),
        gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="DDIM ETA", value=0.0, visible=True),
        gr.Slider(minimum=1, maximum=50, step=1, label='Sampling Iterations', value=2),
        gr.Slider(minimum=1, maximum=16, step=1, label='Samples Per Iteration', value=2),
        gr.Slider(minimum=1.0, maximum=20.0, step=0.5, label='Classifier-free Guidance Scale', value=5.0),
        gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label='Denoising Strength', value=0.75),
        gr.Textbox(label="Seed (blank to randomize)", lines=1, value=""),
        gr.Slider(minimum=0, maximum=2048, step=64, label="Resize Width (0 to auto-adjust to nearest 64 multiple)", value=0),
        gr.Slider(minimum=0, maximum=2048, step=64, label="Resize Height (0 to auto-adjust to nearest 64 multiple)", value=0),
        gr.Slider(minimum=0, maximum=20, step=1, label="Latent Channels", value=4, visible=False),
        gr.HTML(notes_translation, visible=False),
    ],
    outputs=[
        gr.Gallery(),
        gr.HTML(label='Notes'),
    ],
    title="Stable Diffusion Image-to-Image K2",
    description="Generate images from images with Stable Diffusion",
    allow_flagging='never',
)

demo = gr.TabbedInterface(interface_list=[dream_interface, img2img_interface], tab_names=["Dream", "Image Translation"])
demo.queue(concurrency_count=1)
demo.launch(show_error=True, server_name='0.0.0.0')

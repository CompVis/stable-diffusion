"""
clone/install the following repo beforehand
git clone https://github.com/deforum/stable-diffusion
git clone https://github.com/deforum/k-diffusion
pip install -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers
pip install -e git+https://github.com/openai/CLIP.git@main#egg=clip

weights for openai/clip-vit-large-patch14 and stable-diffusion sd-v1-4.ckpt are downloaded to ./weights
in ./stable-diffusion/ldm/modules/encoders/modules.py, load from local weights (local_files_only=True) for FrozenCLIPEmbedder()
"""

import os
from typing import Optional, List
from collections import OrderedDict
from PIL import Image
from itertools import islice
import shutil
import json
from IPython import display
import argparse, glob, os, pathlib, subprocess, sys, time
import cv2
import numpy as np
import pandas as pd
import random
import requests
import shutil
import torch
import torch.nn as nn
from torch import autocast
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.utils import make_grid
from contextlib import contextmanager, nullcontext
from einops import rearrange, repeat
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from skimage.exposure import match_histograms
from tqdm import tqdm, trange
from types import SimpleNamespace
import subprocess
from base64 import b64encode
from cog import BasePredictor, Input, Path

sys.path.append("./src/taming-transformers")
sys.path.append("./src/clip")
sys.path.append("./stable-diffusion/")
sys.path.append("./k-diffusion")

from helpers import save_samples, sampler_fn
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

from k_diffusion import sampling
from k_diffusion.external import CompVisDenoiser


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        ckpt_config_path = (
            "./stable-diffusion/configs/stable-diffusion/v1-inference.yaml"
        )
        ckpt_path = "./weights/sd-v1-4.ckpt"
        local_config = OmegaConf.load(f"{ckpt_config_path}")

        half_precision = True
        self.model = load_model_from_config(
            local_config, f"{ckpt_path}", half_precision=half_precision
        )
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.model = self.model.to(self.device)

    def predict(
        self,
        max_frames: int = Input(
            description="Number of frames for animation", ge=100, le=1000, default=30
        ),
        animation_prompts: str = Input(
            default="0: a beautiful portrait of a woman by Artgerm, trending on Artstation",
            description="Prompt for animation. Provide 'frame number : prompt at this frame', separate different prompts with '|'. Make sure the frame number does not exceed the max_frames.",
        ),
        angle: str = Input(
            description="angle parameter for the motion", default="0:(0)"
        ),
        zoom: str = Input(
            description="zoom parameter for the motion", default="0: (1.04)"
        ),
        translation_x: str = Input(
            description="translation_x parameter for the motion", default="0: (0)"
        ),
        translation_y: str = Input(
            description="translation_y parameter for the motion", default="0: (0)"
        ),
        color_coherence: str = Input(
            choices=[
                "None",
                "Match Frame 0 HSV",
                "Match Frame 0 LAB",
                "Match Frame 0 RGB",
            ],
            default="Match Frame 0 LAB",
        ),
        sampler: str = Input(
            choices=[
                "klms",
                "dpm2",
                "dpm2_ancestral",
                "heun",
                "euler",
                "euler_ancestral",
                "plms",
                "ddim",
            ],
            default="plms",
        ),
        fps: int = Input(
            default=15, ge=10, le=60, description="Choose fps for the video."
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> Path:
        """Run a single prediction on the model"""

        # sanity checks:
        animation_prompts_dict = {}
        animation_prompts = animation_prompts.split("|")
        assert len(animation_prompts) > 0, "Please provide valid prompt for animation."
        if len(animation_prompts) == 1:
            animation_prompts = {0: animation_prompts[0]}
        else:
            for frame_prompt in animation_prompts:
                frame_prompt = frame_prompt.split(":")
                assert (
                    len(frame_prompt) == 2
                ), "Please follow the 'frame_num: prompt' format."
                frame_id, prompt = frame_prompt[0].strip(), frame_prompt[1].strip()
                assert (
                    frame_id.isdigit() and 0<= int(frame_id) <= max_frames
                ), "frame_num should be an integer and 0<= frame_num <= max_frames"
                assert (
                    int(frame_id) not in animation_prompts_dict
                ), f"Duplicate prompts for frame_num {frame_id}. "
                assert len(prompt) > 0, "prompt cannot be empty"
                animation_prompts_dict[int(frame_id)] = prompt
            animation_prompts = OrderedDict(sorted(animation_prompts_dict.items()))

        outdir = "cog_out"
        if os.path.exists(outdir):
            shutil.rmtree(outdir)
        os.makedirs(outdir)

        # load default args
        anim_args = SimpleNamespace(**DeforumAnimArgs())

        # overwrite with user input
        anim_args.max_frames = max_frames
        anim_args.angle = angle
        anim_args.zoom = zoom
        anim_args.translation_x = translation_x
        anim_args.translation_y = translation_y
        anim_args.color_coherence = color_coherence

        if anim_args.animation_mode == "None":
            anim_args.max_frames = 1

        if anim_args.key_frames:
            anim_args.angle_series = get_inbetweens(
                anim_args, parse_key_frames(anim_args.angle)
            )
            anim_args.zoom_series = get_inbetweens(
                anim_args, parse_key_frames(anim_args.zoom)
            )
            anim_args.translation_x_series = get_inbetweens(
                anim_args, parse_key_frames(anim_args.translation_x)
            )
            anim_args.translation_y_series = get_inbetweens(
                anim_args, parse_key_frames(anim_args.translation_y)
            )
            anim_args.noise_schedule_series = get_inbetweens(
                anim_args, parse_key_frames(anim_args.noise_schedule)
            )
            anim_args.strength_schedule_series = get_inbetweens(
                anim_args, parse_key_frames(anim_args.strength_schedule)
            )
            anim_args.contrast_schedule_series = get_inbetweens(
                anim_args, parse_key_frames(anim_args.contrast_schedule)
            )

        args = SimpleNamespace(**DeforumArgs())
        args.timestring = time.strftime("%Y%m%d%H%M%S")
        args.strength = max(0.0, min(1.0, args.strength))

        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")
        args.seed = seed
        args.outdir = outdir

        if anim_args.animation_mode == "Video Input":
            args.use_init = True
        if not args.use_init:
            args.init_image = None
            args.strength = 0
        if args.sampler == "plms" and (
            args.use_init or anim_args.animation_mode != "None"
        ):
            print(f"Init images aren't supported with PLMS yet, switching to KLMS")
            args.sampler = "klms"
        if args.sampler != "ddim":
            args.ddim_eta = 0

        if anim_args.animation_mode == "2D":
            anim_args.animation_prompts = animation_prompts
            render_animation(args, anim_args, self.model, self.device)
        elif anim_args.animation_mode == "Video Input":
            render_input_video(args, anim_args, self.model, self.device)
        elif anim_args.animation_mode == "Interpolation":
            render_interpolation(args, anim_args, self.model, self.device)
        else:
            render_image_batch(args, prompts, self.model, self.device)

        # make video
        image_path = os.path.join(args.outdir, f"{args.timestring}_%05d.png")
        mp4_path = f"/tmp/out.mp4"

        # make video
        cmd = [
            "ffmpeg",
            "-y",
            "-vcodec",
            "png",
            "-r",
            str(fps),
            "-start_number",
            str(0),
            "-i",
            image_path,
            "-frames:v",
            str(anim_args.max_frames),
            "-c:v",
            "libx264",
            "-vf",
            f"fps={fps}",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "17",
            "-preset",
            "veryfast",
            mp4_path,
        ]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            print(stderr)
            raise RuntimeError(stderr)

        return Path(mp4_path)


def DeforumArgs():
    # Save & Display Settings
    batch_name = "StableFun"
    outdir = "cog_output"
    save_settings = False
    save_samples = True
    display_samples = False

    # Image Settings
    n_samples = 1  # hidden
    W = 512
    H = 512
    W, H = map(lambda x: x - x % 64, (W, H))  # resize to integer multiple of 64

    # Init Settings
    use_init = False
    strength = 0.5
    init_image = "https://cdn.pixabay.com/photo/2022/07/30/13/10/green-longhorn-beetle-7353749_1280.jpg"

    # Sampling Settings
    seed = -1
    sampler = "klms"
    steps = 50
    scale = 7
    ddim_eta = 0.0
    dynamic_threshold = None
    static_threshold = None

    # Batch Settings
    n_batch = 1
    seed_behavior = "iter"

    # Grid Settings
    make_grid = False
    grid_rows = 2

    precision = "autocast"
    fixed_code = True
    C = 4
    f = 8

    prompt = ""
    timestring = ""
    init_latent = None
    init_sample = None
    init_c = None

    return locals()


def DeforumAnimArgs():

    # Animation
    animation_mode = "2D"
    max_frames = 1000
    border = "wrap"

    # Motion Parameters
    key_frames = True
    interp_spline = "Linear"
    angle = "0:(0)"
    zoom = "0: (1.04)"
    translation_x = "0: (0)"
    translation_y = "0: (0)"
    noise_schedule = "0: (0.02)"
    strength_schedule = "0: (0.65)"
    contrast_schedule = "0: (1.0)"

    # Coherence
    color_coherence = "Match Frame 0 LAB"

    # Video Input
    video_init_path = "/content/video_in.mp4"
    extract_nth_frame = 1

    # Interpolation
    interpolate_key_frames = False
    interpolate_x_frames = 4

    # Resume Animation
    resume_from_timestring = False
    resume_timestring = " "
    return locals()


def load_model_from_config(
    config, ckpt, verbose=False, device="cuda", half_precision=True
):
    map_location = "cuda"
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

    if half_precision:
        model = model.half().to(device)
    else:
        model = model.to(device)
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


def add_noise(sample: torch.Tensor, noise_amt: float):
    return sample + torch.randn(sample.shape, device=sample.device) * noise_amt


def load_img(path, shape):
    if path.startswith("http://") or path.startswith("https://"):
        image = Image.open(requests.get(path, stream=True).raw).convert("RGB")
    else:
        image = Image.open(path).convert("RGB")

    image = image.resize(shape, resample=Image.LANCZOS)
    image = np.array(image).astype(np.float16) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0


def maintain_colors(prev_img, color_match_sample, mode):
    if mode == "Match Frame 0 RGB":
        return match_histograms(prev_img, color_match_sample, multichannel=True)
    elif mode == "Match Frame 0 HSV":
        prev_img_hsv = cv2.cvtColor(prev_img, cv2.COLOR_RGB2HSV)
        color_match_hsv = cv2.cvtColor(color_match_sample, cv2.COLOR_RGB2HSV)
        matched_hsv = match_histograms(prev_img_hsv, color_match_hsv, multichannel=True)
        return cv2.cvtColor(matched_hsv, cv2.COLOR_HSV2RGB)
    else:  # Match Frame 0 LAB
        prev_img_lab = cv2.cvtColor(prev_img, cv2.COLOR_RGB2LAB)
        color_match_lab = cv2.cvtColor(color_match_sample, cv2.COLOR_RGB2LAB)
        matched_lab = match_histograms(prev_img_lab, color_match_lab, multichannel=True)
        return cv2.cvtColor(matched_lab, cv2.COLOR_LAB2RGB)


def make_callback(sampler, dynamic_threshold=None, static_threshold=None):
    # Creates the callback function to be passed into the samplers
    # The callback function is applied to the image after each step
    def dynamic_thresholding_(img, threshold):
        # Dynamic thresholding from Imagen paper (May 2022)
        s = np.percentile(np.abs(img.cpu()), threshold, axis=tuple(range(1, img.ndim)))
        s = np.max(np.append(s, 1.0))
        torch.clamp_(img, -1 * s, s)
        torch.FloatTensor.div_(img, s)

    # Callback for samplers in the k-diffusion repo, called thus:
    #   callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
    def k_callback(args_dict):
        if static_threshold is not None:
            torch.clamp_(args_dict["x"], -1 * static_threshold, static_threshold)
        if dynamic_threshold is not None:
            dynamic_thresholding_(args_dict["x"], dynamic_threshold)

    # Function that is called on the image (img) and step (i) at each step
    def img_callback(img, i):
        # Thresholding functions
        if dynamic_threshold is not None:
            dynamic_thresholding_(img, dynamic_threshold)
        if static_threshold is not None:
            torch.clamp_(img, -1 * static_threshold, static_threshold)

    if sampler in ["plms", "ddim"]:
        # Callback function formated for compvis latent diffusion samplers
        callback = img_callback
    else:
        # Default callback function uses k-diffusion sampler variables
        callback = k_callback

    return callback


def generate(
    args, model, device, return_latent=False, return_sample=False, return_c=False
):
    seed_everything(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    if args.sampler == "plms":
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    model_wrap = CompVisDenoiser(model)
    batch_size = args.n_samples
    prompt = args.prompt
    assert prompt is not None
    data = [batch_size * [prompt]]

    init_latent = None
    if args.init_latent is not None:
        init_latent = args.init_latent
    elif args.init_sample is not None:
        init_latent = model.get_first_stage_encoding(
            model.encode_first_stage(args.init_sample)
        )
    elif args.init_image != None and args.init_image != "":
        init_image = load_img(args.init_image, shape=(args.W, args.H)).to(device)
        init_image = repeat(init_image, "1 ... -> b ...", b=batch_size)
        init_latent = model.get_first_stage_encoding(
            model.encode_first_stage(init_image)
        )  # move to latent space

    sampler.make_schedule(
        ddim_num_steps=args.steps, ddim_eta=args.ddim_eta, verbose=False
    )

    t_enc = int((1.0 - args.strength) * args.steps)

    start_code = None
    if args.fixed_code and init_latent == None:
        start_code = torch.randn(
            [args.n_samples, args.C, args.H // args.f, args.W // args.f], device=device
        )

    callback = make_callback(
        sampler=args.sampler,
        dynamic_threshold=args.dynamic_threshold,
        static_threshold=args.static_threshold,
    )

    results = []
    precision_scope = autocast if args.precision == "autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                for prompts in data:
                    uc = None
                    if args.scale != 1.0:
                        uc = model.get_learned_conditioning(batch_size * [""])
                    if isinstance(prompts, tuple):
                        prompts = list(prompts)
                    c = model.get_learned_conditioning(prompts)

                    if args.init_c != None:
                        c = args.init_c

                    if args.sampler in [
                        "klms",
                        "dpm2",
                        "dpm2_ancestral",
                        "heun",
                        "euler",
                        "euler_ancestral",
                    ]:
                        samples = sampler_fn(
                            c=c,
                            uc=uc,
                            args=args,
                            model_wrap=model_wrap,
                            init_latent=init_latent,
                            t_enc=t_enc,
                            device=device,
                            cb=callback,
                        )
                    else:

                        if init_latent != None:
                            z_enc = sampler.stochastic_encode(
                                init_latent,
                                torch.tensor([t_enc] * batch_size).to(device),
                            )
                            samples = sampler.decode(
                                z_enc,
                                c,
                                t_enc,
                                unconditional_guidance_scale=args.scale,
                                unconditional_conditioning=uc,
                            )
                        else:
                            if args.sampler == "plms" or args.sampler == "ddim":
                                shape = [args.C, args.H // args.f, args.W // args.f]
                                samples, _ = sampler.sample(
                                    S=args.steps,
                                    conditioning=c,
                                    batch_size=args.n_samples,
                                    shape=shape,
                                    verbose=False,
                                    unconditional_guidance_scale=args.scale,
                                    unconditional_conditioning=uc,
                                    eta=args.ddim_eta,
                                    x_T=start_code,
                                    img_callback=callback,
                                )

                    if return_latent:
                        results.append(samples.clone())

                    x_samples = model.decode_first_stage(samples)
                    if return_sample:
                        results.append(x_samples.clone())

                    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                    if return_c:
                        results.append(c.clone())

                    for x_sample in x_samples:
                        x_sample = 255.0 * rearrange(
                            x_sample.cpu().numpy(), "c h w -> h w c"
                        )
                        image = Image.fromarray(x_sample.astype(np.uint8))
                        results.append(image)
    return results


def sample_from_cv2(sample: np.ndarray) -> torch.Tensor:
    sample = ((sample.astype(float) / 255.0) * 2) - 1
    sample = sample[None].transpose(0, 3, 1, 2).astype(np.float16)
    sample = torch.from_numpy(sample)
    return sample


def sample_to_cv2(sample: torch.Tensor) -> np.ndarray:
    sample_f32 = rearrange(sample.squeeze().cpu().numpy(), "c h w -> h w c").astype(
        np.float32
    )
    sample_f32 = ((sample_f32 * 0.5) + 0.5).clip(0, 1)
    sample_int8 = (sample_f32 * 255).astype(np.uint8)
    return sample_int8


def make_xform_2d(width, height, translation_x, translation_y, angle, scale):
    center = (width // 2, height // 2)
    trans_mat = np.float32([[1, 0, translation_x], [0, 1, translation_y]])
    rot_mat = cv2.getRotationMatrix2D(center, angle, scale)
    trans_mat = np.vstack([trans_mat, [0, 0, 1]])
    rot_mat = np.vstack([rot_mat, [0, 0, 1]])
    return np.matmul(rot_mat, trans_mat)


def parse_key_frames(string, prompt_parser=None):
    import re

    pattern = r"((?P<frame>[0-9]+):[\s]*[\(](?P<param>[\S\s]*?)[\)])"
    frames = dict()
    for match_object in re.finditer(pattern, string):
        frame = int(match_object.groupdict()["frame"])
        param = match_object.groupdict()["param"]
        if prompt_parser:
            frames[frame] = prompt_parser(param)
        else:
            frames[frame] = param
    if frames == {} and len(string) != 0:
        raise RuntimeError("Key Frame string not correctly formatted")
    return frames


def get_inbetweens(anim_args, key_frames, integer=False):
    key_frame_series = pd.Series([np.nan for a in range(anim_args.max_frames)])

    for i, value in key_frames.items():
        key_frame_series[i] = value
    key_frame_series = key_frame_series.astype(float)

    interp_method = anim_args.interp_spline
    if interp_method == "Cubic" and len(key_frames.items()) <= 3:
        interp_method = "Quadratic"
    if interp_method == "Quadratic" and len(key_frames.items()) <= 2:
        interp_method = "Linear"

    key_frame_series[0] = key_frame_series[key_frame_series.first_valid_index()]
    key_frame_series[anim_args.max_frames - 1] = key_frame_series[
        key_frame_series.last_valid_index()
    ]
    key_frame_series = key_frame_series.interpolate(
        method=interp_method.lower(), limit_direction="both"
    )
    if integer:
        return key_frame_series.astype(int)
    return key_frame_series


def next_seed(args):
    if args.seed_behavior == "iter":
        args.seed += 1
    elif args.seed_behavior == "fixed":
        pass  # always keep seed the same
    else:
        args.seed = random.randint(0, 2**32)
    return args.seed


def render_image_batch(args, prompts, model, device):
    args.prompts = prompts

    # create output folder for the batch
    os.makedirs(args.outdir, exist_ok=True)
    if args.save_settings or args.save_samples:
        print(f"Saving to {os.path.join(args.outdir, args.timestring)}_*")

    # save settings for the batch
    if args.save_settings:
        filename = os.path.join(args.outdir, f"{args.timestring}_settings.txt")
        with open(filename, "w+", encoding="utf-8") as f:
            json.dump(dict(args.__dict__), f, ensure_ascii=False, indent=4)

    index = 0

    # function for init image batching
    init_array = []
    if args.use_init:
        if args.init_image == "":
            raise FileNotFoundError("No path was given for init_image")
        if args.init_image.startswith("http://") or args.init_image.startswith(
            "https://"
        ):
            init_array.append(args.init_image)
        elif not os.path.isfile(args.init_image):
            if (
                args.init_image[-1] != "/"
            ):  # avoids path error by adding / to end if not there
                args.init_image += "/"
            for image in sorted(
                os.listdir(args.init_image)
            ):  # iterates dir and appends images to init_array
                if image.split(".")[-1] in ("png", "jpg", "jpeg"):
                    init_array.append(args.init_image + image)
        else:
            init_array.append(args.init_image)
    else:
        init_array = [""]

    # when doing large batches don't flood browser with images
    clear_between_batches = args.n_batch >= 32

    for iprompt, prompt in enumerate(prompts):
        args.prompt = prompt

        all_images = []

        for batch_index in range(args.n_batch):
            if clear_between_batches:
                display.clear_output(wait=True)
            print(f"Batch {batch_index+1} of {args.n_batch}")

            for image in init_array:  # iterates the init images
                args.init_image = image
                results = generate(args, model, device)
                for image in results:
                    if args.make_grid:
                        all_images.append(T.functional.pil_to_tensor(image))
                    if args.save_samples:
                        filename = f"{args.timestring}_{index:05}_{args.seed}.png"
                        image.save(os.path.join(args.outdir, filename))
                    if args.display_samples:
                        display.display(image)
                    index += 1
                args.seed = next_seed(args)

        # print(len(all_images))
        if args.make_grid:
            grid = make_grid(all_images, nrow=int(len(all_images) / args.grid_rows))
            grid = rearrange(grid, "c h w -> h w c").cpu().numpy()
            filename = f"{args.timestring}_{iprompt:05d}_grid_{args.seed}.png"
            grid_image = Image.fromarray(grid.astype(np.uint8))
            grid_image.save(os.path.join(args.outdir, filename))
            display.clear_output(wait=True)
            display.display(grid_image)


def render_animation(args, anim_args, model, device):
    # animations use key framed prompts
    args.prompts = anim_args.animation_prompts

    # resume animation
    start_frame = 0
    if anim_args.resume_from_timestring:
        for tmp in os.listdir(args.outdir):
            if tmp.split("_")[0] == anim_args.resume_timestring:
                start_frame += 1
        start_frame = start_frame - 1

    # create output folder for the batch
    os.makedirs(args.outdir, exist_ok=True)
    print(f"Saving animation frames to {args.outdir}")

    # save settings for the batch
    settings_filename = os.path.join(args.outdir, f"{args.timestring}_settings.txt")

    # resume from timestring
    if anim_args.resume_from_timestring:
        args.timestring = anim_args.resume_timestring

    # expand prompts out to per-frame
    prompt_series = pd.Series([np.nan for a in range(anim_args.max_frames)])
    for i, prompt in anim_args.animation_prompts.items():
        prompt_series[i] = prompt
    prompt_series = prompt_series.ffill().bfill()

    # check for video inits
    using_vid_init = anim_args.animation_mode == "Video Input"

    args.n_samples = 1
    prev_sample = None
    color_match_sample = None
    for frame_idx in range(start_frame, anim_args.max_frames):
        print(f"Rendering animation frame {frame_idx} of {anim_args.max_frames}")

        # resume animation
        if anim_args.resume_from_timestring:
            path = os.path.join(args.outdir, f"{args.timestring}_{frame_idx-1:05}.png")
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            prev_sample = sample_from_cv2(img)

        # apply transforms to previous frame
        if prev_sample is not None:
            if anim_args.key_frames:
                angle = anim_args.angle_series[frame_idx]
                zoom = anim_args.zoom_series[frame_idx]
                translation_x = anim_args.translation_x_series[frame_idx]
                translation_y = anim_args.translation_y_series[frame_idx]
                noise = anim_args.noise_schedule_series[frame_idx]
                strength = anim_args.strength_schedule_series[frame_idx]
                contrast = anim_args.contrast_schedule_series[frame_idx]
                print(
                    f"angle: {angle}",
                    f"zoom: {zoom}",
                    f"translation_x: {translation_x}",
                    f"translation_y: {translation_y}",
                    f"noise: {noise}",
                    f"strength: {strength}",
                    f"contrast: {contrast}",
                )
            xform = make_xform_2d(
                args.W, args.H, translation_x, translation_y, angle, zoom
            )

            # transform previous frame
            prev_img = sample_to_cv2(prev_sample)
            prev_img = cv2.warpPerspective(
                prev_img,
                xform,
                (prev_img.shape[1], prev_img.shape[0]),
                borderMode=cv2.BORDER_WRAP
                if anim_args.border == "wrap"
                else cv2.BORDER_REPLICATE,
            )

            # apply color matching
            if anim_args.color_coherence != "None":
                if color_match_sample is None:
                    color_match_sample = prev_img.copy()
                else:
                    prev_img = maintain_colors(
                        prev_img, color_match_sample, anim_args.color_coherence
                    )

            # apply scaling
            contrast_sample = prev_img * contrast
            # apply frame noising
            noised_sample = add_noise(sample_from_cv2(contrast_sample), noise)

            # use transformed previous frame as init for current
            args.use_init = True
            args.init_sample = noised_sample.half().to(device)
            args.strength = max(0.0, min(1.0, strength))

        # grab prompt for current frame
        args.prompt = prompt_series[frame_idx]
        print(f"{args.prompt} {args.seed}")

        # grab init image for current frame
        if using_vid_init:
            init_frame = os.path.join(
                args.outdir, "inputframes", f"{frame_idx+1:04}.jpg"
            )
            print(f"Using video init frame {init_frame}")
            args.init_image = init_frame

        # sample the diffusion model
        results = generate(args, model, device, return_latent=False, return_sample=True)
        sample, image = results[0], results[1]

        filename = f"{args.timestring}_{frame_idx:05}.png"
        image.save(os.path.join(args.outdir, filename))
        if not using_vid_init:
            prev_sample = sample

        display.clear_output(wait=True)
        display.display(image)

        args.seed = next_seed(args)


def render_input_video(args, anim_args, model, dvice):
    # create a folder for the video input frames to live in
    video_in_frame_path = os.path.join(args.outdir, "inputframes")
    os.makedirs(os.path.join(args.outdir, video_in_frame_path), exist_ok=True)

    # save the video frames from input video
    print(
        f"Exporting Video Frames (1 every {anim_args.extract_nth_frame}) frames to {video_in_frame_path}..."
    )
    try:
        for f in pathlib.Path(video_in_frame_path).glob("*.jpg"):
            f.unlink()
    except:
        pass
    vf = r"select=not(mod(n\," + str(anim_args.extract_nth_frame) + "))"
    subprocess.run(
        [
            "ffmpeg",
            "-i",
            f"{anim_args.video_init_path}",
            "-vf",
            f"{vf}",
            "-vsync",
            "vfr",
            "-q:v",
            "2",
            "-loglevel",
            "error",
            "-stats",
            os.path.join(video_in_frame_path, "%04d.jpg"),
        ],
        stdout=subprocess.PIPE,
    ).stdout.decode("utf-8")

    # determine max frames from length of input frames
    anim_args.max_frames = len(
        [f for f in pathlib.Path(video_in_frame_path).glob("*.jpg")]
    )

    args.use_init = True
    print(
        f"Loading {anim_args.max_frames} input frames from {video_in_frame_path} and saving video frames to {args.outdir}"
    )
    render_animation(args, anim_args, model, device)


def render_interpolation(args, anim_args, model, device):
    # animations use key framed prompts
    args.prompts = animation_prompts

    # create output folder for the batch
    os.makedirs(args.outdir, exist_ok=True)
    print(f"Saving animation frames to {args.outdir}")

    # save settings for the batch
    settings_filename = os.path.join(args.outdir, f"{args.timestring}_settings.txt")
    # with open(settings_filename, "w+", encoding="utf-8") as f:
    #     s = {**dict(args.__dict__), **dict(anim_args.__dict__)}
    #     json.dump(s, f, ensure_ascii=False, indent=4)

    # Interpolation Settings
    args.n_samples = 1
    args.seed_behavior = (
        "fixed"  # force fix seed at the moment bc only 1 seed is available
    )
    prompts_c_s = []  # cache all the text embeddings

    print(f"Preparing for interpolation of the following...")

    for i, prompt in animation_prompts.items():
        args.prompt = prompt

        # sample the diffusion model
        results = generate(args, model, device, return_c=True)
        c, image = results[0], results[1]
        prompts_c_s.append(c)

        # display.clear_output(wait=True)
        display.display(image)

        args.seed = next_seed(args)

    display.clear_output(wait=True)
    print(f"Interpolation start...")

    frame_idx = 0

    if anim_args.interpolate_key_frames:
        for i in range(len(prompts_c_s) - 1):
            dist_frames = (
                list(animation_prompts.items())[i + 1][0]
                - list(animation_prompts.items())[i][0]
            )
            if dist_frames <= 0:
                print("key frames duplicated or reversed. interpolation skipped.")
                return
            else:
                for j in range(dist_frames):
                    # interpolate the text embedding
                    prompt1_c = prompts_c_s[i]
                    prompt2_c = prompts_c_s[i + 1]
                    args.init_c = prompt1_c.add(
                        prompt2_c.sub(prompt1_c).mul(j * 1 / dist_frames)
                    )

                    # sample the diffusion model
                    results = generate(args, model, device)
                    image = results[0]

                    filename = f"{args.timestring}_{frame_idx:05}.png"
                    image.save(os.path.join(args.outdir, filename))
                    frame_idx += 1

                    display.clear_output(wait=True)
                    display.display(image)

                    args.seed = next_seed(args)

    else:
        for i in range(len(prompts_c_s) - 1):
            for j in range(anim_args.interpolate_x_frames + 1):
                # interpolate the text embedding
                prompt1_c = prompts_c_s[i]
                prompt2_c = prompts_c_s[i + 1]
                args.init_c = prompt1_c.add(
                    prompt2_c.sub(prompt1_c).mul(
                        j * 1 / (anim_args.interpolate_x_frames + 1)
                    )
                )

                # sample the diffusion model
                results = generate(args, model, device)
                image = results[0]

                filename = f"{args.timestring}_{frame_idx:05}.png"
                image.save(os.path.join(args.outdir, filename))
                frame_idx += 1

                display.clear_output(wait=True)
                display.display(image)

                args.seed = next_seed(args)

    # generate the last prompt
    args.init_c = prompts_c_s[-1]
    results = generate(args, model, device)
    image = results[0]
    filename = f"{args.timestring}_{frame_idx:05}.png"
    image.save(os.path.join(args.outdir, filename))

    display.clear_output(wait=True)
    display.display(image)
    args.seed = next_seed(args)

    # clear init_c
    args.init_c = None

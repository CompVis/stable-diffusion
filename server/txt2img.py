import argparse, os, sys, glob
import cv2
import torch
import boto3
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from imwatermark import WatermarkEncoder
from itertools import islice
from einops import rearrange
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext
from random import randint
from uuid import uuid4
from datauri import DataURI
import base64
from io import BytesIO

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor


# load safety model
safety_model_id = "CompVis/stable-diffusion-safety-checker"
safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images

def upload_to_s3(filepath, filename, folder):
    """Uploads image to s3 bucket"""
    bucket = f"{os.environ.get('S3_BUCKET')}"

    s3 = boto3.client(
        service_name= 's3',
        aws_access_key_id= os.environ.get('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key= os.environ.get('AWS_SECRET_ACCESS_KEY'),
    )

    s3.upload_file(
        Filename=filepath,
        Bucket=f"{os.environ.get('S3_BUCKET')}",
        Key=filename, 
        ExtraArgs={
            'ContentType': 'image/png',
        }
    )

    return f"https://{bucket}.s3.us-east-2.amazonaws.com/{filename}"


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


def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img


ckpt = "models/ldm/stable-diffusion-v1/model.ckpt"
config = "configs/stable-diffusion/v1-inference.yaml"
config = OmegaConf.load(f"{config}")
model = load_model_from_config(config, f"{ckpt}")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = model.to(device)


def generate(prompt, **kwargs):
    seed = kwargs.get('seed', 0)
    width = kwargs.get('width', 512)
    height = kwargs.get('height', 512)
    steps = kwargs.get('steps', 30)
    iterations = kwargs.get('iterations', 1)
    scale = kwargs.get('scale', 7.5)

    print(prompt)

    outdir = 'outputs\samples'
    factor = 8 # downsampling factor
    channels = 4
    precision = 'autocast'
    skip_grid = False
    skip_save = False
    plms = True
    n_rows = 0
    fixed_code = False
    ddim_eta = 0.0
    timer = time.time()

    results = []

    seed = randint(1, 1_000_000) if seed == 0 else seed
    seed_everything(seed)

    if plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    os.makedirs(outdir, exist_ok=True)
    outpath = outdir

    print("Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")
    wm = "StableDiffusionV1"
    wm_encoder = WatermarkEncoder()
    wm_encoder.set_watermark('bytes', wm.encode('utf-8'))

    batch_size = 1
    n_rows = n_rows if n_rows > 0 else batch_size

    assert prompt is not None
    data = [batch_size * [prompt]]

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))

    start_code = None
    if fixed_code:
        start_code = torch.randn([batch_size, channels, height // factor, width // factor], device=device)


    precision_scope = autocast if precision=="autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                tic = time.time()
                all_samples = list()
                for n in trange(iterations, desc="Sampling"):
                    for prompts in tqdm(data, desc="data"):
                        uc = None
                        if scale != 1.0:
                            uc = model.get_learned_conditioning(batch_size * [""])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        c = model.get_learned_conditioning(prompts)
                        shape = [channels, height // factor, width // factor]
                        samples_ddim, _ = sampler.sample(S=steps,
                                                         conditioning=c,
                                                         batch_size=batch_size,
                                                         shape=shape,
                                                         verbose=False,
                                                         unconditional_guidance_scale=scale,
                                                         unconditional_conditioning=uc,
                                                         eta=ddim_eta,
                                                         x_T=start_code)

                        x_samples_ddim = model.decode_first_stage(samples_ddim)
                        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                        x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

                        x_image_torch = torch.from_numpy(x_samples_ddim).permute(0, 3, 1, 2)

                        for x_sample in x_image_torch:
                            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                            filename = f"{base_count}_{seed}_{uuid4().hex}.png"
                            filepath = os.path.join(sample_path, filename) 
                            img = Image.fromarray(x_sample.astype(np.uint8))
                            img = put_watermark(img, wm_encoder)
                            img.save(filepath)

                            file_url = upload_to_s3(filepath, filename, 'samples')

                            result = {
                                'seed': seed,
                                'image': file_url,
                            }

                            results.append(result)
                            base_count += 1
                            seed += 1

                toc = time.time()
    return results, time.time() - timer

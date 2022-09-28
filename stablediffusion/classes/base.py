import argparse
import os
import torch
import numpy as np
import cv2
import logging
from PIL import Image
from einops import rearrange
from imwatermark import WatermarkEncoder
from torch import autocast
from pytorch_lightning import seed_everything  # FAILS
from omegaconf import OmegaConf
from contextlib import  nullcontext
# import common classes from stable diffusion
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.util import instantiate_from_config
from itertools import islice
# load safety model
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor
import logger as log
HOME = os.path.expanduser("~")
safety_model_id = f"{HOME}/stablediffusion/models/CompVis/stable-diffusion-safety-checker"
#safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
log.info("SETTING UP StableDiffusionSafetyChecker FROM PRETRAINED")
safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)

# import txt2img functions from stable diffusion
def load_model_from_config(config, ckpt, verbose=False):
    log.info(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        log.info(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        log.error("missing keys:")
        log.error(m)
    if len(u) > 0 and verbose:
        log.error("unexpected keys:")
        log.error(u)

    model.cuda().half()
    model.eval()
    return model


def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img


def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y)/255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


class BaseModel:
    """
    Base model class which is inherited by model classes (see classes.txt2img.py and classes.img2img.py)
    :attribute args: list of arguments to be parsed by argparse, set this in each child class
    :attribute opt: parsed arguments, set by parse_arguments()
    :attribute parser: argparse parser, set by parse_arguments()
    :attribute config: config file, set by load_config()
    :attribute data: data to be used for sampling, set by prepare_data()
    :attribute batch_size: batch size, set by prepare_data()
    :attribute n_rows: number of rows in the output image, set by prepare_data()
    :attribute outpath: output path, set by initialize_outdir()
    :attribute sampler: sampler object, set by initialize_sampler()
    :attribute device: device to use, set by load_model()
    :attribute model: model object, set by load_model()
    :attribute wm_encoder: watermark encoder object, set by initialize_watermark()
    :attribute base_count: base count, set by initialize_base_count()
    :attribute grid_count: grid count, set by initialize_grid_count()
    :attribute sample_path: sample path, set by create_sample_path()
    :attribute start_code: start code, set by initialize_start_code()
    :attribute precision_scope: precision scope, set by set_precision_scope()
    :attribute initialized: whether the model has been initialized, set by initialize()
    """
    args = []

    def __init__(self, *args, **kwargs):
        self.opt = {}
        self.initialize_logging()
        self.parser = None
        self.config = None
        self.data = None
        self.batch_size = None
        self.n_rows = None
        self.outpath = None
        self.device = kwargs.get("device", None)
        self.model = kwargs.get("model", None)
        self.wm_encoder = None
        self.base_count = None
        self.grid_count = None
        self.sample_path = None
        self.start_code = None
        self.precision_scope = None
        self.initialized = False
        self.init_model(kwargs.get("options", {}))


    def initialize_logging(self):
        # create path and file if not exist
        HOME = os.path.expanduser("~")
        path = f"{HOME}/stablediffusion/stablediffusiond.log"
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        if not os.path.exists(path):
            open(path, 'a').close()
        self.log = logging.getLogger(__name__)
        self.log.setLevel(logging.INFO)
        self.log.addHandler(logging.FileHandler(path))

    @property
    def plms_sampler(self):
        return PLMSSampler(self.model)

    @property
    def ddim_sampler(self):
        return DDIMSampler(self.model)

    def initialize(self):
        """
        Initialize the model
        :return:
        """
        if self.initialized:
            return
        self.initialized = True
        self.load_config()
        if not self.model or not self.device:
            self.load_model()
        self.initialize_outdir()
        self.initialize_watermark()
        self.create_sample_path()
        self.initialize_start_code()

    def parse_arguments(self):
        """
        Parse arguments, the arguments are defined by each child class
        :return:
        """
        parser = argparse.ArgumentParser()
        for arg in self.args:
            parser.add_argument(
                f'--{arg["arg"]}',
                **{k: v for k, v in arg.items() if k != "arg"}
            )
        self.parser = parser
        self.opt = self.parser.parse_args()

    def parse_options(self, options):
        """
        Parse options
        :param options: options to parse
        :return:
        """
        for opt in options:
            self.opt.__setattr__(opt[0], opt[1])

    def initialize_options(self):
        """
        Initialize options, by default check for laion400m and set the corresponding options
        :return:
        """
        if self.opt.__contains__("laion400m") and self.opt.laion400m:
            self.log.info("Falling back to LAION 400M model...")
            self.opt.config = "configs/latent-diffusion/txt2img-1p4B-eval.yaml"
            self.opt.ckpt = "models/ldm/text2img-large/model.ckpt"
            self.opt.outdir = "outputs/txt2img-samples-laion400m"

    def set_seed(self):
        """
        Seed everything using the current seed.
        This allows us to re-seed the model with a new seed that can remain static or be modified, e.g. when sampling.
        :return:
        """
        seed_everything(self.opt.seed)

    def load_config(self):
        """
        Load config file
        :return:
        """
        self.config = OmegaConf.load(f"{self.opt.config}")

    def load_model(self):
        """
        Load the stable diffusion model
        :return:
        """
        self.model = load_model_from_config(self.config, f"{self.opt.ckpt}")
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = self.model.to(self.device)

    def initialize_outdir(self):
        """"
        Initialize the output directory
        :return:
        """
        try:
            os.makedirs(self.opt.outdir, exist_ok=True)
        except Exception:
            pass
        self.outpath = self.opt.outdir

    def initialize_watermark(self):
        """
        Initialize the watermark encoder
        :return:
        """
        self.log.info("Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")
        wm = "StableDiffusionV1"
        self.wm_encoder = WatermarkEncoder()
        self.wm_encoder.set_watermark('bytes', wm.encode('utf-8'))

    def numpy_to_pil(self, images):
        """
        Convert a numpy image or a batch of images to a PIL image.
        """
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]
        return pil_images

    def check_safety(self, x_image):
        # safety_checker_input = safety_feature_extractor(self.numpy_to_pil(x_image), return_tensors="pt")
        x_checked_image, has_nsfw_concept = safety_checker(images=x_image, clip_input=safety_checker_input.pixel_values)
        assert x_checked_image.shape[0] == len(has_nsfw_concept)
        for i in range(len(has_nsfw_concept)):
            if has_nsfw_concept[i]:
                x_checked_image[i] = load_replacement(x_checked_image[i])
        return x_checked_image, has_nsfw_concept

    def filter_nsfw_content(self, x_samples_ddim):
        """
        Check if the samples are safe for work, replace them with a placeholder if not
        :param x_samples_ddim:
        :return:
        """
        if self.opt.do_nsfw_filter:
            x_samples_ddim, has_nsfw = self.check_safety(x_samples_ddim)
            return x_samples_ddim
        return x_samples_ddim

    def add_watermark(self, img):
        """
        Adds digital watermark to image
        :param img:
        :return:
        """
        if self.opt.do_watermark != "":
            img = put_watermark(img, self.wm_encoder)
        return img

    def prepare_data(self):
        """
        Prepare data for sampling
        :return:
        """
        batch_size = self.opt.n_samples if "n_samples" in self.opt else 1
        n_rows = self.opt.n_rows if (
                "n_rows" in self.opt and self.opt.n_rows > 0) else 0
        from_file = self.opt.from_file if "from_file" in self.opt else False
        if not from_file:
            prompt = self.opt.prompt if "prompt" in self.opt else None
            data = [batch_size * [prompt]]

        else:
            self.log.info(f"reading prompts from {from_file}")
            with open(from_file, "r") as f:
                data = f.read().splitlines()
                data = list(chunk(data, batch_size))
        self.n_rows = n_rows
        self.batch_size = batch_size
        self.data = data

    def create_sample_path(self):
        """
        Create the sample path
        :return:
        """
        sample_path = self.outpath
        os.makedirs(sample_path, exist_ok=True)
        self.sample_path = sample_path

    def initialize_start_code(self):
        """
        Initialize the start based on fixed_code settings
        :return:
        """
        if self.opt.fixed_code:
            self.start_code = torch.randn([
                self.opt.n_samples,
                self.opt.C,
                self.opt.H // self.opt.f,
                self.opt.W // self.opt.f
            ], device=self.device)

    def set_precision_scope(self):
        """
        Define the precision scope
        :return:
        """
        self.precision_scope = autocast if self.opt.precision=="autocast" else nullcontext

    def get_first_stage_sample(self, model, samples):
        samples_ddim = model.decode_first_stage(samples)
        return torch.clamp((samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

    def init_model(self, options):
        self.parse_arguments()
        self.set_seed()
        self.initialize_options()
        if options:
            self.parse_options(options)
        self.prepare_data()
        self.initialize()

    def sample(self, options=None):
        """
        Sample from the model
        :param options:
        :return:
        """
        self.init_model(options)
        self.set_precision_scope()
        self.base_count = len(os.listdir(self.sample_path))
        self.grid_count = len(os.listdir(self.outpath)) - 1

    def save_image(self, samples, sample_path, base_count, watermark=True):
        """
        Save the image
        :param x_checked_image_torch:
        :param sample_path:
        :param base_count:
        :return:
        """
        for x_sample in samples:
            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
            img = Image.fromarray(x_sample.astype(np.uint8))
            img = self.add_watermark(img)
            file_name = os.path.join(sample_path, f"{base_count:05}.png")
            img.save(file_name)
            return file_name

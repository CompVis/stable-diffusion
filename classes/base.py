import argparse
import sys
import os
import torch
from torch import autocast
from pytorch_lightning import seed_everything
from omegaconf import OmegaConf
from contextlib import  nullcontext

# import common classes from stable diffusion
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

# import txt2img functions from stable diffusion
from scripts.txt2img import load_model_from_config
from scripts.txt2img import chunk


class BaseModel:
    args = []

    def __init__(self, *args, **kwargs):
        self.opt = {}
        self.config = None
        self.parser = None
        self.data = None
        self.batch_size = None
        self.n_rows = None
        self.outpath = None
        self.sampler = None
        self.device = None
        self.model = None
        self.wm_encoder = None
        self.base_count = None
        self.grid_count = None
        self.sample_path = None
        self.start_code = None
        self.precision_scope = None
        self.initialized = False

    def initialize(self):
        if self.initialized:
            return
        self.initialized = True
        self.load_config()
        self.load_model()
        self.initialize_sampler()
        self.initialize_outdir()
        self.initialize_watermark()
        self.create_sample_path()
        self.initialize_start_code()

    def parse_arguments(self):
        parser = argparse.ArgumentParser()
        for arg in self.args:
            parser.add_argument(
                f'--{arg["arg"]}',
                **{k: v for k, v in arg.items() if k != "arg"}
            )
        self.parser = parser
        self.opt = self.parser.parse_args()

    def parse_options(self, options):
        for opt in options:
            if opt[0] in self.opt:
                self.opt.__setattr__(opt[0], opt[1])

    def initialize_options(self):
        if self.opt.laion400m:
            print("Falling back to LAION 400M model...")
            self.opt.config = "configs/latent-diffusion/txt2img-1p4B-eval.yaml"
            self.opt.ckpt = "models/ldm/text2img-large/model.ckpt"
            self.opt.outdir = "outputs/txt2img-samples-laion400m"

    def set_seed(self):
        seed_everything(self.opt.seed)

    def load_config(self):
        self.config = OmegaConf.load(f"{self.opt.config}")

    def load_model(self):
        self.model = load_model_from_config(self.config, f"{self.opt.ckpt}")
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = self.model.to(self.device)

    def initialize_sampler(self):
        if self.opt.plms:
            self.sampler = PLMSSampler(self.model)
        else:
            self.sampler = DDIMSampler(self.model)

    def initialize_outdir(self):
        os.makedirs(self.opt.outdir, exist_ok=True)
        self.outpath = self.opt.outdir

    def initialize_watermark(self):
        print("Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")
        # wm = "StableDiffusionV1"
        # wm_encoder = WatermarkEncoder()
        # wm_encoder.set_watermark('bytes', wm.encode('utf-8'))

    def prepare_data(self):
        batch_size = self.opt.n_samples
        n_rows = self.opt.n_rows if self.opt.n_rows > 0 else batch_size
        if not self.opt.from_file:
            prompt = self.opt.prompt
            assert prompt is not None
            data = [batch_size * [prompt]]

        else:
            print(f"reading prompts from {self.opt.from_file}")
            with open(self.opt.from_file, "r") as f:
                data = f.read().splitlines()
                data = list(chunk(data, batch_size))
        self.n_rows = n_rows
        self.batch_size = batch_size
        self.data = data

    def create_sample_path(self):
        sample_path = os.path.join(self.outpath, os.path.join("samples", self.opt.out_folder))
        os.makedirs(sample_path, exist_ok=True)
        self.sample_path = sample_path

    def initialize_start_code(self):
        if self.opt.fixed_code:
            self.start_code = torch.randn([
                self.opt.n_samples,
                self.opt.C,
                self.opt.H // self.opt.f,
                self.opt.W // self.opt.f
            ], device=self.device)

    def set_precision_scope(self):
        self.precision_scope = autocast if self.opt.precision=="autocast" else nullcontext

    def sample(self, options):
        print("SAMPLING from model wrapper")
        self.parse_arguments()
        self.initialize_options()
        self.parse_options(options)
        self.set_seed()
        self.set_precision_scope()
        self.prepare_data()
        self.initialize()
        self.base_count = len(os.listdir(self.sample_path))
        self.grid_count = len(os.listdir(self.outpath)) - 1

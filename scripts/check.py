print("Checking python environment")

# Import core libraries
import os, re, time, sys, asyncio, ctypes, math
import torch
import scipy
import numpy as np
from random import randint
from omegaconf import OmegaConf
from PIL import Image
from itertools import islice, product
from einops import rearrange, repeat
from pytorch_lightning import seed_everything
from contextlib import nullcontext
from typing import Optional
from safetensors.torch import load_file

# Import built libraries
from ldm.util import instantiate_from_config
from optimUtils import split_weighted_subprompts
from autoencoder.pixelvae import load_pixelvae_model
import tomesd

# Import PyTorch functions
from torch import autocast
from torch import Tensor
from torch.nn import functional as F
from torch.nn.modules.utils import _pair

# Import logging libraries
import traceback, warnings
import logging as pylog
from transformers import logging

# Import websocket tools
import requests
from websockets import serve, connect
from io import BytesIO

# Import post-processing libraries
import hitherdither
from rembg import remove

# Import console management libraries
import pygetwindow as gw
from rich import print as rprint
from colorama import just_fix_windows_console

print("Python environment initialized successfully")
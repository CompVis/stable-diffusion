#!/usr/bin/env python

# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path

import torch

from ...utils import is_xpu_available
from .config_args import ClusterConfig, default_json_config_file
from .config_utils import SubcommandHelpFormatter


description = "Create a default config file for Accelerate with only a few flags set."


def write_basic_config(mixed_precision="no", save_location: str = default_json_config_file, use_xpu: bool = False):
    """
    Creates and saves a basic cluster config to be used on a local machine with potentially multiple GPUs. Will also
    set CPU if it is a CPU-only machine.

    Args:
        mixed_precision (`str`, *optional*, defaults to "no"):
            Mixed Precision to use. Should be one of "no", "fp16", or "bf16"
        save_location (`str`, *optional*, defaults to `default_json_config_file`):
            Optional custom save location. Should be passed to `--config_file` when using `accelerate launch`. Default
            location is inside the huggingface cache folder (`~/.cache/huggingface`) but can be overriden by setting
            the `HF_HOME` environmental variable, followed by `accelerate/default_config.yaml`.
        use_xpu (`bool`, *optional*, defaults to `False`):
            Whether to use XPU if available.
    """
    path = Path(save_location)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        print(
            f"Configuration already exists at {save_location}, will not override. Run `accelerate config` manually or pass a different `save_location`."
        )
        return False
    mixed_precision = mixed_precision.lower()
    if mixed_precision not in ["no", "fp16", "bf16", "fp8"]:
        raise ValueError(
            f"`mixed_precision` should be one of 'no', 'fp16', 'bf16', or 'fp8'. Received {mixed_precision}"
        )
    config = {
        "compute_environment": "LOCAL_MACHINE",
        "mixed_precision": mixed_precision,
    }
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        config["num_processes"] = num_gpus
        config["use_cpu"] = False
        if num_gpus > 1:
            config["distributed_type"] = "MULTI_GPU"
        else:
            config["distributed_type"] = "NO"
    elif is_xpu_available() and use_xpu:
        num_xpus = torch.xpu.device_count()
        config["num_processes"] = num_xpus
        config["use_cpu"] = False
        if num_xpus > 1:
            config["distributed_type"] = "MULTI_XPU"
        else:
            config["distributed_type"] = "NO"
    else:
        num_xpus = 0
        config["use_cpu"] = True
        config["num_processes"] = 1
        config["distributed_type"] = "NO"
    config = ClusterConfig(**config)
    config.to_json_file(path)
    return path


def default_command_parser(parser, parents):
    parser = parser.add_parser("default", parents=parents, help=description, formatter_class=SubcommandHelpFormatter)
    parser.add_argument(
        "--config_file",
        default=default_json_config_file,
        help=(
            "The path to use to store the config file. Will default to a file named default_config.yaml in the cache "
            "location, which is the content of the environment `HF_HOME` suffixed with 'accelerate', or if you don't have "
            "such an environment variable, your cache directory ('~/.cache' or the content of `XDG_CACHE_HOME`) suffixed "
            "with 'huggingface'."
        ),
        dest="save_location",
    )

    parser.add_argument(
        "--mixed_precision",
        choices=["no", "fp16", "bf16"],
        type=str,
        help="Whether or not to use mixed precision training. "
        "Choose between FP16 and BF16 (bfloat16) training. "
        "BF16 training is only supported on Nvidia Ampere GPUs and PyTorch 1.10 or later.",
        default="no",
    )
    parser.set_defaults(func=default_config_command)
    return parser


def default_config_command(args):
    config_file = write_basic_config(args.mixed_precision, args.save_location)
    if config_file:
        print(f"accelerate configuration saved at {config_file}")

#!/usr/bin/env python

# Copyright 2022 The HuggingFace Team. All rights reserved.
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

import argparse
import os
import subprocess

from packaging.version import Version, parse

from accelerate.commands.config.config_args import default_config_file, load_config_from_file


_description = "Run commands across TPU VMs for initial setup before running `accelerate launch`."


def tpu_command_parser(subparsers=None):
    if subparsers is not None:
        parser = subparsers.add_parser("tpu-config", description=_description)
    else:
        parser = argparse.ArgumentParser("Accelerate tpu-config command", description=_description)
    # Core arguments
    config_args = parser.add_argument_group(
        "Config Arguments", "Arguments that can be configured through `accelerate config`."
    )
    config_args.add_argument(
        "--config_file",
        type=str,
        default=None,
        help="Path to the config file to use for accelerate.",
    )
    config_args.add_argument(
        "--tpu_name",
        default=None,
        help="The name of the TPU to use. If not specified, will use the TPU specified in the config file.",
    )
    config_args.add_argument(
        "--tpu_zone",
        default=None,
        help="The zone of the TPU to use. If not specified, will use the zone specified in the config file.",
    )
    pod_args = parser.add_argument_group("TPU Arguments", "Arguments for options ran inside the TPU.")
    pod_args.add_argument(
        "--use_alpha",
        action="store_true",
        help="Whether to use `gcloud alpha` when running the TPU training script instead of `gcloud`.",
    )
    pod_args.add_argument(
        "--command_file",
        default=None,
        help="The path to the file containing the commands to run on the pod on startup.",
    )
    pod_args.add_argument(
        "--command",
        action="append",
        nargs="+",
        help="A command to run on the pod. Can be passed multiple times.",
    )
    pod_args.add_argument(
        "--install_accelerate",
        action="store_true",
        help="Whether to install accelerate on the pod. Defaults to False.",
    )
    pod_args.add_argument(
        "--accelerate_version",
        default="latest",
        help="The version of accelerate to install on the pod. If not specified, will use the latest pypi version. Specify 'dev' to install from GitHub.",
    )
    pod_args.add_argument(
        "--debug", action="store_true", help="If set, will print the command that would be run instead of running it."
    )

    if subparsers is not None:
        parser.set_defaults(func=tpu_command_launcher)
    return parser


def tpu_command_launcher(args):
    defaults = None

    # Get the default from the config file if it exists.
    if args.config_file is not None or os.path.isfile(default_config_file):
        defaults = load_config_from_file(args.config_file)
        if not args.command_file and defaults.command_file is not None and not args.command:
            args.command_file = defaults.command_file
        if not args.command and defaults.commands is not None:
            args.command = defaults.commands
        if not args.tpu_name:
            args.tpu_name = defaults.tpu_name
        if not args.tpu_zone:
            args.tpu_zone = defaults.tpu_zone
    if args.accelerate_version == "dev":
        args.accelerate_version = "git+https://github.com/huggingface/accelerate.git"
    elif args.accelerate_version == "latest":
        args.accelerate_version = "accelerate -U"
    elif isinstance(parse(args.accelerate_version), Version):
        args.accelerate_version = f"accelerate=={args.accelerate_version}"

    if not args.command_file and not args.command:
        raise ValueError("You must specify either a command file or a command to run on the pod.")

    if args.command_file:
        with open(args.command_file, "r") as f:
            args.command = [f.read().splitlines()]

    # To turn list of lists into list of strings
    if isinstance(args.command[0], list):
        args.command = [line for cmd in args.command for line in cmd]
    # Default to the shared folder and install accelerate
    new_cmd = ["cd /usr/share"]
    if args.install_accelerate:
        new_cmd += [f"pip install {args.accelerate_version}"]
    new_cmd += args.command
    args.command = "; ".join(new_cmd)

    # Then send it to gcloud
    # Eventually try to use google-api-core to do this instead of subprocess
    cmd = ["gcloud"]
    if args.use_alpha:
        cmd += ["alpha"]
    cmd += [
        "compute",
        "tpus",
        "tpu-vm",
        "ssh",
        args.tpu_name,
        "--zone",
        args.tpu_zone,
        "--command",
        args.command,
        "--worker",
        "all",
    ]
    if args.debug:
        print(f"Running {' '.join(cmd)}")
        return
    subprocess.run(cmd)
    print("Successfully setup pod.")


def main():
    parser = tpu_command_parser()
    args = parser.parse_args()

    tpu_command_launcher(args)

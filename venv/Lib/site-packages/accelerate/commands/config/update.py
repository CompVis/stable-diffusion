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

from pathlib import Path

from .config_args import default_config_file, load_config_from_file
from .config_utils import SubcommandHelpFormatter


description = "Update an existing config file with the latest defaults while maintaining the old configuration."


def update_config(args):
    """
    Update an existing config file with the latest defaults while maintaining the old configuration.
    """
    config_file = args.config_file
    if config_file is None and Path(default_config_file).exists():
        config_file = default_config_file
    elif not Path(config_file).exists():
        raise ValueError(f"The passed config file located at {config_file} doesn't exist.")
    config = load_config_from_file(config_file)

    if config_file.endswith(".json"):
        config.to_json_file(config_file)
    else:
        config.to_yaml_file(config_file)
    return config_file


def update_command_parser(parser, parents):
    parser = parser.add_parser("update", parents=parents, help=description, formatter_class=SubcommandHelpFormatter)
    parser.add_argument(
        "--config_file",
        default=None,
        help=(
            "The path to the config file to update. Will default to a file named default_config.yaml in the cache "
            "location, which is the content of the environment `HF_HOME` suffixed with 'accelerate', or if you don't have "
            "such an environment variable, your cache directory ('~/.cache' or the content of `XDG_CACHE_HOME`) suffixed "
            "with 'huggingface'."
        ),
    )

    parser.set_defaults(func=update_config_command)
    return parser


def update_config_command(args):
    config_file = update_config(args)
    print(f"Sucessfully updated the configuration file at {config_file}.")

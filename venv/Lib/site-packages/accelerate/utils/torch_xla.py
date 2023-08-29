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

import subprocess
import sys

import pkg_resources


def install_xla(upgrade: bool = False):
    """
    Helper function to install appropriate xla wheels based on the `torch` version in Google Colaboratory.

    Args:
        upgrade (`bool`, *optional*, defaults to `False`):
            Whether to upgrade `torch` and install the latest `torch_xla` wheels.

    Example:

    ```python
    >>> from accelerate.utils import install_xla

    >>> install_xla(upgrade=True)
    ```
    """
    in_colab = False
    if "IPython" in sys.modules:
        in_colab = "google.colab" in str(sys.modules["IPython"].get_ipython())

    if in_colab:
        if upgrade:
            torch_install_cmd = ["pip", "install", "-U", "torch"]
            subprocess.run(torch_install_cmd, check=True)
        # get the current version of torch
        torch_version = pkg_resources.get_distribution("torch").version
        torch_version_trunc = torch_version[: torch_version.rindex(".")]
        xla_wheel = f"https://storage.googleapis.com/tpu-pytorch/wheels/colab/torch_xla-{torch_version_trunc}-cp37-cp37m-linux_x86_64.whl"
        xla_install_cmd = ["pip", "install", xla_wheel]
        subprocess.run(xla_install_cmd, check=True)
    else:
        raise RuntimeError("`install_xla` utility works only on google colab.")

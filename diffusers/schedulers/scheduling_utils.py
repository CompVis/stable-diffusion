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
from dataclasses import dataclass
from typing import Union

import numpy as np
import torch

from ..utils import BaseOutput


SCHEDULER_CONFIG_NAME = "scheduler_config.json"


@dataclass
class SchedulerOutput(BaseOutput):
    """
    Base class for the scheduler's step function output.

    Args:
        prev_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample (x_{t-1}) of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
    """

    prev_sample: torch.FloatTensor


class SchedulerMixin:
    """
    Mixin containing common functions for the schedulers.
    """

    config_name = SCHEDULER_CONFIG_NAME
    ignore_for_config = ["tensor_format"]

    def set_format(self, tensor_format="pt"):
        self.tensor_format = tensor_format
        if tensor_format == "pt":
            for key, value in vars(self).items():
                if isinstance(value, np.ndarray):
                    setattr(self, key, torch.from_numpy(value))

        return self

    def clip(self, tensor, min_value=None, max_value=None):
        tensor_format = getattr(self, "tensor_format", "pt")

        if tensor_format == "np":
            return np.clip(tensor, min_value, max_value)
        elif tensor_format == "pt":
            return torch.clamp(tensor, min_value, max_value)

        raise ValueError(f"`self.tensor_format`: {self.tensor_format} is not valid.")

    def log(self, tensor):
        tensor_format = getattr(self, "tensor_format", "pt")

        if tensor_format == "np":
            return np.log(tensor)
        elif tensor_format == "pt":
            return torch.log(tensor)

        raise ValueError(f"`self.tensor_format`: {self.tensor_format} is not valid.")

    def match_shape(self, values: Union[np.ndarray, torch.Tensor], broadcast_array: Union[np.ndarray, torch.Tensor]):
        """
        Turns a 1-D array into an array or tensor with len(broadcast_array.shape) dims.

        Args:
            values: an array or tensor of values to extract.
            broadcast_array: an array with a larger shape of K dimensions with the batch
                dimension equal to the length of timesteps.
        Returns:
            a tensor of shape [batch_size, 1, ...] where the shape has K dims.
        """

        tensor_format = getattr(self, "tensor_format", "pt")
        values = values.flatten()

        while len(values.shape) < len(broadcast_array.shape):
            values = values[..., None]
        if tensor_format == "pt":
            values = values.to(broadcast_array.device)

        return values

    def norm(self, tensor):
        tensor_format = getattr(self, "tensor_format", "pt")
        if tensor_format == "np":
            return np.linalg.norm(tensor)
        elif tensor_format == "pt":
            return torch.norm(tensor.reshape(tensor.shape[0], -1), dim=-1).mean()

        raise ValueError(f"`self.tensor_format`: {self.tensor_format} is not valid.")

    def randn_like(self, tensor, generator=None):
        tensor_format = getattr(self, "tensor_format", "pt")
        if tensor_format == "np":
            return np.random.randn(*np.shape(tensor))
        elif tensor_format == "pt":
            # return torch.randn_like(tensor)
            return torch.randn(tensor.shape, layout=tensor.layout, generator=generator).to(tensor.device)

        raise ValueError(f"`self.tensor_format`: {self.tensor_format} is not valid.")

    def zeros_like(self, tensor):
        tensor_format = getattr(self, "tensor_format", "pt")
        if tensor_format == "np":
            return np.zeros_like(tensor)
        elif tensor_format == "pt":
            return torch.zeros_like(tensor)

        raise ValueError(f"`self.tensor_format`: {self.tensor_format} is not valid.")

# Copyright 2022 Google Brain and The HuggingFace Team. All rights reserved.
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

# DISCLAIMER: This file is strongly influenced by https://github.com/yang-song/score_sde_pytorch

# TODO(Patrick, Anton, Suraj) - make scheduler framework indepedent and clean-up a bit

import numpy as np
import torch

from ..configuration_utils import ConfigMixin, register_to_config
from .scheduling_utils import SchedulerMixin


class ScoreSdeVpScheduler(SchedulerMixin, ConfigMixin):
    """
    The variance preserving stochastic differential equation (SDE) scheduler.

    [`~ConfigMixin`] takes care of storing all config attributes that are passed in the scheduler's `__init__`
    function, such as `num_train_timesteps`. They can be accessed via `scheduler.config.num_train_timesteps`.
    [`~ConfigMixin`] also provides general loading and saving functionality via the [`~ConfigMixin.save_config`] and
    [`~ConfigMixin.from_config`] functios.

    For more information, see the original paper: https://arxiv.org/abs/2011.13456

    UNDER CONSTRUCTION

    """

    @register_to_config
    def __init__(self, num_train_timesteps=2000, beta_min=0.1, beta_max=20, sampling_eps=1e-3, tensor_format="np"):

        self.sigmas = None
        self.discrete_sigmas = None
        self.timesteps = None

    def set_timesteps(self, num_inference_steps):
        self.timesteps = torch.linspace(1, self.config.sampling_eps, num_inference_steps)

    def step_pred(self, score, x, t):
        if self.timesteps is None:
            raise ValueError(
                "`self.timesteps` is not set, you need to run 'set_timesteps' after creating the scheduler"
            )

        # TODO(Patrick) better comments + non-PyTorch
        # postprocess model score
        log_mean_coeff = (
            -0.25 * t**2 * (self.config.beta_max - self.config.beta_min) - 0.5 * t * self.config.beta_min
        )
        std = torch.sqrt(1.0 - torch.exp(2.0 * log_mean_coeff))
        score = -score / std[:, None, None, None]

        # compute
        dt = -1.0 / len(self.timesteps)

        beta_t = self.config.beta_min + t * (self.config.beta_max - self.config.beta_min)
        drift = -0.5 * beta_t[:, None, None, None] * x
        diffusion = torch.sqrt(beta_t)
        drift = drift - diffusion[:, None, None, None] ** 2 * score
        x_mean = x + drift * dt

        # add noise
        noise = torch.randn_like(x)
        x = x_mean + diffusion[:, None, None, None] * np.sqrt(-dt) * noise

        return x, x_mean

    def __len__(self):
        return self.config.num_train_timesteps

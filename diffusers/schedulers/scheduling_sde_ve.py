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

import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import BaseOutput
from .scheduling_utils import SchedulerMixin, SchedulerOutput


@dataclass
class SdeVeOutput(BaseOutput):
    """
    Output class for the ScoreSdeVeScheduler's step function output.

    Args:
        prev_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample (x_{t-1}) of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
        prev_sample_mean (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            Mean averaged `prev_sample`. Same as `prev_sample`, only mean-averaged over previous timesteps.
    """

    prev_sample: torch.FloatTensor
    prev_sample_mean: torch.FloatTensor


class ScoreSdeVeScheduler(SchedulerMixin, ConfigMixin):
    """
    The variance exploding stochastic differential equation (SDE) scheduler.

    For more information, see the original paper: https://arxiv.org/abs/2011.13456

    [`~ConfigMixin`] takes care of storing all config attributes that are passed in the scheduler's `__init__`
    function, such as `num_train_timesteps`. They can be accessed via `scheduler.config.num_train_timesteps`.
    [`~ConfigMixin`] also provides general loading and saving functionality via the [`~ConfigMixin.save_config`] and
    [`~ConfigMixin.from_config`] functios.

    Args:
        snr (`float`):
            coefficient weighting the step from the model_output sample (from the network) to the random noise.
        sigma_min (`float`):
                initial noise scale for sigma sequence in sampling procedure. The minimum sigma should mirror the
                distribution of the data.
        sigma_max (`float`): maximum value used for the range of continuous timesteps passed into the model.
        sampling_eps (`float`): the end value of sampling, where timesteps decrease progessively from 1 to
        epsilon.
        correct_steps (`int`): number of correction steps performed on a produced sample.
        tensor_format (`str`): "np" or "pt" for the expected format of samples passed to the Scheduler.
    """

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 2000,
        snr: float = 0.15,
        sigma_min: float = 0.01,
        sigma_max: float = 1348.0,
        sampling_eps: float = 1e-5,
        correct_steps: int = 1,
        tensor_format: str = "pt",
    ):
        # setable values
        self.timesteps = None

        self.set_sigmas(num_train_timesteps, sigma_min, sigma_max, sampling_eps)

        self.tensor_format = tensor_format
        self.set_format(tensor_format=tensor_format)

    def set_timesteps(self, num_inference_steps: int, sampling_eps: float = None):
        """
        Sets the continuous timesteps used for the diffusion chain. Supporting function to be run before inference.

        Args:
            num_inference_steps (`int`):
                the number of diffusion steps used when generating samples with a pre-trained model.
            sampling_eps (`float`, optional): final timestep value (overrides value given at Scheduler instantiation).

        """
        sampling_eps = sampling_eps if sampling_eps is not None else self.config.sampling_eps
        tensor_format = getattr(self, "tensor_format", "pt")
        if tensor_format == "np":
            self.timesteps = np.linspace(1, sampling_eps, num_inference_steps)
        elif tensor_format == "pt":
            self.timesteps = torch.linspace(1, sampling_eps, num_inference_steps)
        else:
            raise ValueError(f"`self.tensor_format`: {self.tensor_format} is not valid.")

    def set_sigmas(
        self, num_inference_steps: int, sigma_min: float = None, sigma_max: float = None, sampling_eps: float = None
    ):
        """
        Sets the noise scales used for the diffusion chain. Supporting function to be run before inference.

        The sigmas control the weight of the `drift` and `diffusion` components of sample update.

        Args:
            num_inference_steps (`int`):
                the number of diffusion steps used when generating samples with a pre-trained model.
            sigma_min (`float`, optional):
                initial noise scale value (overrides value given at Scheduler instantiation).
            sigma_max (`float`, optional): final noise scale value (overrides value given at Scheduler instantiation).
            sampling_eps (`float`, optional): final timestep value (overrides value given at Scheduler instantiation).

        """
        sigma_min = sigma_min if sigma_min is not None else self.config.sigma_min
        sigma_max = sigma_max if sigma_max is not None else self.config.sigma_max
        sampling_eps = sampling_eps if sampling_eps is not None else self.config.sampling_eps
        if self.timesteps is None:
            self.set_timesteps(num_inference_steps, sampling_eps)

        tensor_format = getattr(self, "tensor_format", "pt")
        if tensor_format == "np":
            self.discrete_sigmas = np.exp(np.linspace(np.log(sigma_min), np.log(sigma_max), num_inference_steps))
            self.sigmas = np.array([sigma_min * (sigma_max / sigma_min) ** t for t in self.timesteps])
        elif tensor_format == "pt":
            self.discrete_sigmas = torch.exp(torch.linspace(np.log(sigma_min), np.log(sigma_max), num_inference_steps))
            self.sigmas = torch.tensor([sigma_min * (sigma_max / sigma_min) ** t for t in self.timesteps])
        else:
            raise ValueError(f"`self.tensor_format`: {self.tensor_format} is not valid.")

    def get_adjacent_sigma(self, timesteps, t):
        tensor_format = getattr(self, "tensor_format", "pt")
        if tensor_format == "np":
            return np.where(timesteps == 0, np.zeros_like(t), self.discrete_sigmas[timesteps - 1])
        elif tensor_format == "pt":
            return torch.where(
                timesteps == 0,
                torch.zeros_like(t.to(timesteps.device)),
                self.discrete_sigmas[timesteps - 1].to(timesteps.device),
            )

        raise ValueError(f"`self.tensor_format`: {self.tensor_format} is not valid.")

    def set_seed(self, seed):
        warnings.warn(
            "The method `set_seed` is deprecated and will be removed in version `0.4.0`. Please consider passing a"
            " generator instead.",
            DeprecationWarning,
        )
        tensor_format = getattr(self, "tensor_format", "pt")
        if tensor_format == "np":
            np.random.seed(seed)
        elif tensor_format == "pt":
            torch.manual_seed(seed)
        else:
            raise ValueError(f"`self.tensor_format`: {self.tensor_format} is not valid.")

    def step_pred(
        self,
        model_output: Union[torch.FloatTensor, np.ndarray],
        timestep: int,
        sample: Union[torch.FloatTensor, np.ndarray],
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
        **kwargs,
    ) -> Union[SdeVeOutput, Tuple]:
        """
        Predict the sample at the previous timestep by reversing the SDE. Core function to propagate the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor` or `np.ndarray`): direct output from learned diffusion model.
            timestep (`int`): current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor` or `np.ndarray`):
                current instance of sample being created by diffusion process.
            generator: random number generator.
            return_dict (`bool`): option for returning tuple rather than SchedulerOutput class

        Returns:
            [`~schedulers.scheduling_sde_ve.SdeVeOutput`] or `tuple`: [`~schedulers.scheduling_sde_ve.SdeVeOutput`] if
            `return_dict` is True, otherwise a `tuple`. When returning a tuple, the first element is the sample tensor.

        """
        if "seed" in kwargs and kwargs["seed"] is not None:
            self.set_seed(kwargs["seed"])

        if self.timesteps is None:
            raise ValueError(
                "`self.timesteps` is not set, you need to run 'set_timesteps' after creating the scheduler"
            )

        timestep = timestep * torch.ones(
            sample.shape[0], device=sample.device
        )  # torch.repeat_interleave(timestep, sample.shape[0])
        timesteps = (timestep * (len(self.timesteps) - 1)).long()

        # mps requires indices to be in the same device, so we use cpu as is the default with cuda
        timesteps = timesteps.to(self.discrete_sigmas.device)

        sigma = self.discrete_sigmas[timesteps].to(sample.device)
        adjacent_sigma = self.get_adjacent_sigma(timesteps, timestep).to(sample.device)
        drift = self.zeros_like(sample)
        diffusion = (sigma**2 - adjacent_sigma**2) ** 0.5

        # equation 6 in the paper: the model_output modeled by the network is grad_x log pt(x)
        # also equation 47 shows the analog from SDE models to ancestral sampling methods
        drift = drift - diffusion[:, None, None, None] ** 2 * model_output

        #  equation 6: sample noise for the diffusion term of
        noise = self.randn_like(sample, generator=generator)
        prev_sample_mean = sample - drift  # subtract because `dt` is a small negative timestep
        # TODO is the variable diffusion the correct scaling term for the noise?
        prev_sample = prev_sample_mean + diffusion[:, None, None, None] * noise  # add impact of diffusion field g

        if not return_dict:
            return (prev_sample, prev_sample_mean)

        return SdeVeOutput(prev_sample=prev_sample, prev_sample_mean=prev_sample_mean)

    def step_correct(
        self,
        model_output: Union[torch.FloatTensor, np.ndarray],
        sample: Union[torch.FloatTensor, np.ndarray],
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
        **kwargs,
    ) -> Union[SchedulerOutput, Tuple]:
        """
        Correct the predicted sample based on the output model_output of the network. This is often run repeatedly
        after making the prediction for the previous timestep.

        Args:
            model_output (`torch.FloatTensor` or `np.ndarray`): direct output from learned diffusion model.
            sample (`torch.FloatTensor` or `np.ndarray`):
                current instance of sample being created by diffusion process.
            generator: random number generator.
            return_dict (`bool`): option for returning tuple rather than SchedulerOutput class

        Returns:
            [`~schedulers.scheduling_sde_ve.SdeVeOutput`] or `tuple`: [`~schedulers.scheduling_sde_ve.SdeVeOutput`] if
            `return_dict` is True, otherwise a `tuple`. When returning a tuple, the first element is the sample tensor.

        """
        if "seed" in kwargs and kwargs["seed"] is not None:
            self.set_seed(kwargs["seed"])

        if self.timesteps is None:
            raise ValueError(
                "`self.timesteps` is not set, you need to run 'set_timesteps' after creating the scheduler"
            )

        # For small batch sizes, the paper "suggest replacing norm(z) with sqrt(d), where d is the dim. of z"
        # sample noise for correction
        noise = self.randn_like(sample, generator=generator)

        # compute step size from the model_output, the noise, and the snr
        grad_norm = self.norm(model_output)
        noise_norm = self.norm(noise)
        step_size = (self.config.snr * noise_norm / grad_norm) ** 2 * 2
        step_size = step_size * torch.ones(sample.shape[0]).to(sample.device)
        # self.repeat_scalar(step_size, sample.shape[0])

        # compute corrected sample: model_output term and noise term
        prev_sample_mean = sample + step_size[:, None, None, None] * model_output
        prev_sample = prev_sample_mean + ((step_size * 2) ** 0.5)[:, None, None, None] * noise

        if not return_dict:
            return (prev_sample,)

        return SchedulerOutput(prev_sample=prev_sample)

    def __len__(self):
        return self.config.num_train_timesteps

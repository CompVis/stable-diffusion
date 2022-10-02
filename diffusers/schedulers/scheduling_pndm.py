# Copyright 2022 Zhejiang University Team and The HuggingFace Team. All rights reserved.
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

# DISCLAIMER: This file is strongly influenced by https://github.com/ermongroup/ddim

import math
from typing import Optional, Tuple, Union

import numpy as np
import torch

from ..configuration_utils import ConfigMixin, register_to_config
from .scheduling_utils import SchedulerMixin, SchedulerOutput


def betas_for_alpha_bar(num_diffusion_timesteps, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function, which defines the cumulative product of
    (1-beta) over time from t = [0,1].

    Contains a function alpha_bar that takes an argument t and transforms it to the cumulative product of (1-beta) up
    to that part of the diffusion process.


    Args:
        num_diffusion_timesteps (`int`): the number of betas to produce.
        max_beta (`float`): the maximum beta to use; use values lower than 1 to
                     prevent singularities.

    Returns:
        betas (`np.ndarray`): the betas used by the scheduler to step the model outputs
    """

    def alpha_bar(time_step):
        return math.cos((time_step + 0.008) / 1.008 * math.pi / 2) ** 2

    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas, dtype=np.float32)


class PNDMScheduler(SchedulerMixin, ConfigMixin):
    """
    Pseudo numerical methods for diffusion models (PNDM) proposes using more advanced ODE integration techniques,
    namely Runge-Kutta method and a linear multi-step method.

    [`~ConfigMixin`] takes care of storing all config attributes that are passed in the scheduler's `__init__`
    function, such as `num_train_timesteps`. They can be accessed via `scheduler.config.num_train_timesteps`.
    [`~ConfigMixin`] also provides general loading and saving functionality via the [`~ConfigMixin.save_config`] and
    [`~ConfigMixin.from_config`] functios.

    For more details, see the original paper: https://arxiv.org/abs/2202.09778

    Args:
        num_train_timesteps (`int`): number of diffusion steps used to train the model.
        beta_start (`float`): the starting `beta` value of inference.
        beta_end (`float`): the final `beta` value.
        beta_schedule (`str`):
            the beta schedule, a mapping from a beta range to a sequence of betas for stepping the model. Choose from
            `linear`, `scaled_linear`, or `squaredcos_cap_v2`.
        trained_betas (`np.ndarray`, optional): TODO
        tensor_format (`str`): whether the scheduler expects pytorch or numpy arrays
        skip_prk_steps (`bool`):
            allows the scheduler to skip the Runge-Kutta steps that are defined in the original paper as being required
            before plms steps; defaults to `False`.

    """

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        trained_betas: Optional[np.ndarray] = None,
        tensor_format: str = "pt",
        skip_prk_steps: bool = False,
    ):
        if trained_betas is not None:
            self.betas = np.asarray(trained_betas)
        if beta_schedule == "linear":
            self.betas = np.linspace(beta_start, beta_end, num_train_timesteps, dtype=np.float32)
        elif beta_schedule == "scaled_linear":
            # this schedule is very specific to the latent diffusion model.
            self.betas = np.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=np.float32) ** 2
        elif beta_schedule == "squaredcos_cap_v2":
            # Glide cosine schedule
            self.betas = betas_for_alpha_bar(num_train_timesteps)
        else:
            raise NotImplementedError(f"{beta_schedule} does is not implemented for {self.__class__}")

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas, axis=0)

        self.one = np.array(1.0)

        # For now we only support F-PNDM, i.e. the runge-kutta method
        # For more information on the algorithm please take a look at the paper: https://arxiv.org/pdf/2202.09778.pdf
        # mainly at formula (9), (12), (13) and the Algorithm 2.
        self.pndm_order = 4

        # running values
        self.cur_model_output = 0
        self.counter = 0
        self.cur_sample = None
        self.ets = []

        # setable values
        self.num_inference_steps = None
        self._timesteps = np.arange(0, num_train_timesteps)[::-1].copy()
        self._offset = 0
        self.prk_timesteps = None
        self.plms_timesteps = None
        self.timesteps = None

        self.tensor_format = tensor_format
        self.set_format(tensor_format=tensor_format)

    def set_timesteps(self, num_inference_steps: int, offset: int = 0) -> torch.FloatTensor:
        """
        Sets the discrete timesteps used for the diffusion chain. Supporting function to be run before inference.

        Args:
            num_inference_steps (`int`):
                the number of diffusion steps used when generating samples with a pre-trained model.
            offset (`int`): TODO
        """
        self.num_inference_steps = num_inference_steps
        self._timesteps = list(
            range(0, self.config.num_train_timesteps, self.config.num_train_timesteps // num_inference_steps)
        )
        self._offset = offset
        self._timesteps = np.array([t + self._offset for t in self._timesteps])

        if self.config.skip_prk_steps:
            # for some models like stable diffusion the prk steps can/should be skipped to
            # produce better results. When using PNDM with `self.config.skip_prk_steps` the implementation
            # is based on crowsonkb's PLMS sampler implementation: https://github.com/CompVis/latent-diffusion/pull/51
            self.prk_timesteps = np.array([])
            self.plms_timesteps = np.concatenate([self._timesteps[:-1], self._timesteps[-2:-1], self._timesteps[-1:]])[
                ::-1
            ].copy()
        else:
            prk_timesteps = np.array(self._timesteps[-self.pndm_order :]).repeat(2) + np.tile(
                np.array([0, self.config.num_train_timesteps // num_inference_steps // 2]), self.pndm_order
            )
            self.prk_timesteps = (prk_timesteps[:-1].repeat(2)[1:-1])[::-1].copy()
            self.plms_timesteps = self._timesteps[:-3][
                ::-1
            ].copy()  # we copy to avoid having negative strides which are not supported by torch.from_numpy

        self.timesteps = np.concatenate([self.prk_timesteps, self.plms_timesteps]).astype(np.int64)

        self.ets = []
        self.counter = 0
        self.set_format(tensor_format=self.tensor_format)

    def step(
        self,
        model_output: Union[torch.FloatTensor, np.ndarray],
        timestep: int,
        sample: Union[torch.FloatTensor, np.ndarray],
        return_dict: bool = True,
    ) -> Union[SchedulerOutput, Tuple]:
        """
        Predict the sample at the previous timestep by reversing the SDE. Core function to propagate the diffusion
        process from the learned model outputs (most often the predicted noise).

        This function calls `step_prk()` or `step_plms()` depending on the internal variable `counter`.

        Args:
            model_output (`torch.FloatTensor` or `np.ndarray`): direct output from learned diffusion model.
            timestep (`int`): current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor` or `np.ndarray`):
                current instance of sample being created by diffusion process.
            return_dict (`bool`): option for returning tuple rather than SchedulerOutput class

        Returns:
            [`~schedulers.scheduling_utils.SchedulerOutput`] or `tuple`:
            [`~schedulers.scheduling_utils.SchedulerOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.

        """
        if self.counter < len(self.prk_timesteps) and not self.config.skip_prk_steps:
            return self.step_prk(model_output=model_output, timestep=timestep, sample=sample, return_dict=return_dict)
        else:
            return self.step_plms(model_output=model_output, timestep=timestep, sample=sample, return_dict=return_dict)

    def step_prk(
        self,
        model_output: Union[torch.FloatTensor, np.ndarray],
        timestep: int,
        sample: Union[torch.FloatTensor, np.ndarray],
        return_dict: bool = True,
    ) -> Union[SchedulerOutput, Tuple]:
        """
        Step function propagating the sample with the Runge-Kutta method. RK takes 4 forward passes to approximate the
        solution to the differential equation.

        Args:
            model_output (`torch.FloatTensor` or `np.ndarray`): direct output from learned diffusion model.
            timestep (`int`): current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor` or `np.ndarray`):
                current instance of sample being created by diffusion process.
            return_dict (`bool`): option for returning tuple rather than SchedulerOutput class

        Returns:
            [`~scheduling_utils.SchedulerOutput`] or `tuple`: [`~scheduling_utils.SchedulerOutput`] if `return_dict` is
            True, otherwise a `tuple`. When returning a tuple, the first element is the sample tensor.

        """
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        diff_to_prev = 0 if self.counter % 2 else self.config.num_train_timesteps // self.num_inference_steps // 2
        prev_timestep = max(timestep - diff_to_prev, self.prk_timesteps[-1])
        timestep = self.prk_timesteps[self.counter // 4 * 4]

        if self.counter % 4 == 0:
            self.cur_model_output += 1 / 6 * model_output
            self.ets.append(model_output)
            self.cur_sample = sample
        elif (self.counter - 1) % 4 == 0:
            self.cur_model_output += 1 / 3 * model_output
        elif (self.counter - 2) % 4 == 0:
            self.cur_model_output += 1 / 3 * model_output
        elif (self.counter - 3) % 4 == 0:
            model_output = self.cur_model_output + 1 / 6 * model_output
            self.cur_model_output = 0

        # cur_sample should not be `None`
        cur_sample = self.cur_sample if self.cur_sample is not None else sample

        prev_sample = self._get_prev_sample(cur_sample, timestep, prev_timestep, model_output)
        self.counter += 1

        if not return_dict:
            return (prev_sample,)

        return SchedulerOutput(prev_sample=prev_sample)

    def step_plms(
        self,
        model_output: Union[torch.FloatTensor, np.ndarray],
        timestep: int,
        sample: Union[torch.FloatTensor, np.ndarray],
        return_dict: bool = True,
    ) -> Union[SchedulerOutput, Tuple]:
        """
        Step function propagating the sample with the linear multi-step method. This has one forward pass with multiple
        times to approximate the solution.

        Args:
            model_output (`torch.FloatTensor` or `np.ndarray`): direct output from learned diffusion model.
            timestep (`int`): current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor` or `np.ndarray`):
                current instance of sample being created by diffusion process.
            return_dict (`bool`): option for returning tuple rather than SchedulerOutput class

        Returns:
            [`~scheduling_utils.SchedulerOutput`] or `tuple`: [`~scheduling_utils.SchedulerOutput`] if `return_dict` is
            True, otherwise a `tuple`. When returning a tuple, the first element is the sample tensor.

        """
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        if not self.config.skip_prk_steps and len(self.ets) < 3:
            raise ValueError(
                f"{self.__class__} can only be run AFTER scheduler has been run "
                "in 'prk' mode for at least 12 iterations "
                "See: https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/pipeline_pndm.py "
                "for more information."
            )

        prev_timestep = max(timestep - self.config.num_train_timesteps // self.num_inference_steps, 0)

        if self.counter != 1:
            self.ets.append(model_output)
        else:
            prev_timestep = timestep
            timestep = timestep + self.config.num_train_timesteps // self.num_inference_steps

        if len(self.ets) == 1 and self.counter == 0:
            model_output = model_output
            self.cur_sample = sample
        elif len(self.ets) == 1 and self.counter == 1:
            model_output = (model_output + self.ets[-1]) / 2
            sample = self.cur_sample
            self.cur_sample = None
        elif len(self.ets) == 2:
            model_output = (3 * self.ets[-1] - self.ets[-2]) / 2
        elif len(self.ets) == 3:
            model_output = (23 * self.ets[-1] - 16 * self.ets[-2] + 5 * self.ets[-3]) / 12
        else:
            model_output = (1 / 24) * (55 * self.ets[-1] - 59 * self.ets[-2] + 37 * self.ets[-3] - 9 * self.ets[-4])

        prev_sample = self._get_prev_sample(sample, timestep, prev_timestep, model_output)
        self.counter += 1

        if not return_dict:
            return (prev_sample,)

        return SchedulerOutput(prev_sample=prev_sample)

    def _get_prev_sample(self, sample, timestep, timestep_prev, model_output):
        # See formula (9) of PNDM paper https://arxiv.org/pdf/2202.09778.pdf
        # this function computes x_(t−δ) using the formula of (9)
        # Note that x_t needs to be added to both sides of the equation

        # Notation (<variable name> -> <name in paper>
        # alpha_prod_t -> α_t
        # alpha_prod_t_prev -> α_(t−δ)
        # beta_prod_t -> (1 - α_t)
        # beta_prod_t_prev -> (1 - α_(t−δ))
        # sample -> x_t
        # model_output -> e_θ(x_t, t)
        # prev_sample -> x_(t−δ)
        alpha_prod_t = self.alphas_cumprod[timestep + 1 - self._offset]
        alpha_prod_t_prev = self.alphas_cumprod[timestep_prev + 1 - self._offset]
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        # corresponds to (α_(t−δ) - α_t) divided by
        # denominator of x_t in formula (9) and plus 1
        # Note: (α_(t−δ) - α_t) / (sqrt(α_t) * (sqrt(α_(t−δ)) + sqr(α_t))) =
        # sqrt(α_(t−δ)) / sqrt(α_t))
        sample_coeff = (alpha_prod_t_prev / alpha_prod_t) ** (0.5)

        # corresponds to denominator of e_θ(x_t, t) in formula (9)
        model_output_denom_coeff = alpha_prod_t * beta_prod_t_prev ** (0.5) + (
            alpha_prod_t * beta_prod_t * alpha_prod_t_prev
        ) ** (0.5)

        # full formula (9)
        prev_sample = (
            sample_coeff * sample - (alpha_prod_t_prev - alpha_prod_t) * model_output / model_output_denom_coeff
        )

        return prev_sample

    def add_noise(
        self,
        original_samples: Union[torch.FloatTensor, np.ndarray],
        noise: Union[torch.FloatTensor, np.ndarray],
        timesteps: Union[torch.IntTensor, np.ndarray],
    ) -> torch.Tensor:
        # mps requires indices to be in the same device, so we use cpu as is the default with cuda
        timesteps = timesteps.to(self.alphas_cumprod.device)
        sqrt_alpha_prod = self.alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = self.match_shape(sqrt_alpha_prod, original_samples)
        sqrt_one_minus_alpha_prod = (1 - self.alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = self.match_shape(sqrt_one_minus_alpha_prod, original_samples)

        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples

    def __len__(self):
        return self.config.num_train_timesteps

import copy
import os
import random

import numpy as np
import torch


def enable_full_determinism(seed: int):
    """
    Helper function for reproducible behavior during distributed training. See
    - https://pytorch.org/docs/stable/notes/randomness.html for pytorch
    """
    # set seed first
    set_seed(seed)

    #  Enable PyTorch deterministic mode. This potentially requires either the environment
    #  variable 'CUDA_LAUNCH_BLOCKING' or 'CUBLAS_WORKSPACE_CONFIG' to be set,
    # depending on the CUDA version, so we set them both here
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    torch.use_deterministic_algorithms(True)

    # Enable CUDNN deterministic mode
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_seed(seed: int):
    """
    Args:
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch`.
        seed (`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # ^^ safe to call this function even if cuda is not available


class EMAModel:
    """
    Exponential Moving Average of models weights
    """

    def __init__(
        self,
        model,
        update_after_step=0,
        inv_gamma=1.0,
        power=2 / 3,
        min_value=0.0,
        max_value=0.9999,
        device=None,
    ):
        """
        @crowsonkb's notes on EMA Warmup:
            If gamma=1 and power=1, implements a simple average. gamma=1, power=2/3 are good values for models you plan
            to train for a million or more steps (reaches decay factor 0.999 at 31.6K steps, 0.9999 at 1M steps),
            gamma=1, power=3/4 for models you plan to train for less (reaches decay factor 0.999 at 10K steps, 0.9999
            at 215.4k steps).
        Args:
            inv_gamma (float): Inverse multiplicative factor of EMA warmup. Default: 1.
            power (float): Exponential factor of EMA warmup. Default: 2/3.
            min_value (float): The minimum EMA decay rate. Default: 0.
        """

        self.averaged_model = copy.deepcopy(model).eval()
        self.averaged_model.requires_grad_(False)

        self.update_after_step = update_after_step
        self.inv_gamma = inv_gamma
        self.power = power
        self.min_value = min_value
        self.max_value = max_value

        if device is not None:
            self.averaged_model = self.averaged_model.to(device=device)

        self.decay = 0.0
        self.optimization_step = 0

    def get_decay(self, optimization_step):
        """
        Compute the decay factor for the exponential moving average.
        """
        step = max(0, optimization_step - self.update_after_step - 1)
        value = 1 - (1 + step / self.inv_gamma) ** -self.power

        if step <= 0:
            return 0.0

        return max(self.min_value, min(value, self.max_value))

    @torch.no_grad()
    def step(self, new_model):
        ema_state_dict = {}
        ema_params = self.averaged_model.state_dict()

        self.decay = self.get_decay(self.optimization_step)

        for key, param in new_model.named_parameters():
            if isinstance(param, dict):
                continue
            try:
                ema_param = ema_params[key]
            except KeyError:
                ema_param = param.float().clone() if param.ndim == 1 else copy.deepcopy(param)
                ema_params[key] = ema_param

            if not param.requires_grad:
                ema_params[key].copy_(param.to(dtype=ema_param.dtype).data)
                ema_param = ema_params[key]
            else:
                ema_param.mul_(self.decay)
                ema_param.add_(param.data.to(dtype=ema_param.dtype), alpha=1 - self.decay)

            ema_state_dict[key] = ema_param

        for key, param in new_model.named_buffers():
            ema_state_dict[key] = param

        self.averaged_model.load_state_dict(ema_state_dict, strict=False)
        self.optimization_step += 1

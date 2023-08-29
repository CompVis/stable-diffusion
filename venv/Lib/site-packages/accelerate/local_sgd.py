# Copyright 2023 The HuggingFace Team. All rights reserved.
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
import torch

from accelerate import Accelerator, DistributedType


class LocalSGD:
    """
    A helper class to support local SGD on top of Accelerator. It simply runs a given number of updates independently
    on each device, and averages model weights every K synchronization step.

    It should be used only in the multi-GPU (or multi-CPU) setup without extensions such as DeepSpeed. In particular,
    this is a simple implementation that cannot support scenarios such as model parallelism.


    Although we are not aware of the true origins of this simple approach, the idea of local SGD is quite old and goes
    back to at least:

    Zhang, J., De Sa, C., Mitliagkas, I., & Ré, C. (2016). [Parallel SGD: When does averaging help?. arXiv preprint
    arXiv:1606.07365.](https://arxiv.org/abs/1606.07365)

    We credit the term Local SGD to the following paper (but there might be earlier references we are not aware of).

    Stich, Sebastian Urban. ["Local SGD Converges Fast and Communicates Little." ICLR 2019-International Conference on
    Learning Representations. No. CONF. 2019.](https://arxiv.org/abs/1805.09767)

    """

    def __enter__(self):
        if self.enabled:
            self.model_sync_obj = self.model.no_sync()
            self.model_sync_obj.__enter__()

        return self

    def __exit__(self, type, value, tb):
        if self.enabled:
            # Average all models on exit
            self._sync_and_avg_model_params()
            self.model_sync_obj.__exit__(type, value, tb)

    def __init__(self, accelerator: Accelerator, model: torch.nn.Module, local_sgd_steps: int, enabled: bool = True):
        """
        Constructor.

        Args:
            model (`torch.nn.Module):
                The model whose parameters we need to average.
            accelerator (`Accelerator`):
                Accelerator object.
            local_sgd_steps (`int`):
                A number of local SGD steps (before model parameters are synchronized).
            enabled (`bool):
                Local SGD is disabled if this parameter set to `False`.
        """
        if accelerator.distributed_type not in [
            DistributedType.NO,
            DistributedType.MULTI_CPU,
            DistributedType.MULTI_GPU,
        ]:
            raise NotImplementedError("LocalSGD is supported only for CPUs and GPUs (no DeepSpeed or MegatronLM)")
        self.enabled = enabled and accelerator.distributed_type != DistributedType.NO
        self.num_steps = 0
        if self.enabled:
            self.accelerator = accelerator
            self.model = model
            self.local_sgd_steps = local_sgd_steps

    def step(self):
        """
        This function makes a "step" and synchronizes model parameters if necessary.
        """
        self.num_steps += 1
        if not self.enabled:
            return

        if self.num_steps % self.local_sgd_steps == 0:
            self._sync_and_avg_model_params()

    def _sync_and_avg_model_params(self):
        """
        Synchronize + Average model parameters across all GPUs
        """

        self.accelerator.wait_for_everyone()
        with self.accelerator.autocast():
            for param in self.model.parameters():
                param.data = self.accelerator.reduce(param.data, reduction="mean")

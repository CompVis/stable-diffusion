#!/usr/bin/env python3
import warnings
from typing import Optional, Tuple, Union

import torch

from ...models import UNet2DModel
from ...pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from ...schedulers import ScoreSdeVeScheduler


class ScoreSdeVePipeline(DiffusionPipeline):
    r"""
    Parameters:
    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)
        unet ([`UNet2DModel`]): U-Net architecture to denoise the encoded image. scheduler ([`SchedulerMixin`]):
            The [`ScoreSdeVeScheduler`] scheduler to be used in combination with `unet` to denoise the encoded image.
    """
    unet: UNet2DModel
    scheduler: ScoreSdeVeScheduler

    def __init__(self, unet: UNet2DModel, scheduler: DiffusionPipeline):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        num_inference_steps: int = 2000,
        generator: Optional[torch.Generator] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        **kwargs,
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        Args:
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            generator (`torch.Generator`, *optional*):
                A [torch generator](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation
                deterministic.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `nd.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipeline_utils.ImagePipelineOutput`] instead of a plain tuple.

        Returns:
            [`~pipeline_utils.ImagePipelineOutput`] or `tuple`: [`~pipelines.utils.ImagePipelineOutput`] if
            `return_dict` is True, otherwise a `tuple. When returning a tuple, the first element is a list with the
            generated images.
        """

        if "torch_device" in kwargs:
            device = kwargs.pop("torch_device")
            warnings.warn(
                "`torch_device` is deprecated as an input argument to `__call__` and will be removed in v0.3.0."
                " Consider using `pipe.to(torch_device)` instead."
            )

            # Set device as before (to be removed in 0.3.0)
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.to(device)

        img_size = self.unet.config.sample_size
        shape = (batch_size, 3, img_size, img_size)

        model = self.unet

        sample = torch.randn(*shape, generator=generator) * self.scheduler.config.sigma_max
        sample = sample.to(self.device)

        self.scheduler.set_timesteps(num_inference_steps)
        self.scheduler.set_sigmas(num_inference_steps)

        for i, t in enumerate(self.progress_bar(self.scheduler.timesteps)):
            sigma_t = self.scheduler.sigmas[i] * torch.ones(shape[0], device=self.device)

            # correction step
            for _ in range(self.scheduler.correct_steps):
                model_output = self.unet(sample, sigma_t).sample
                sample = self.scheduler.step_correct(model_output, sample, generator=generator).prev_sample

            # prediction step
            model_output = model(sample, sigma_t).sample
            output = self.scheduler.step_pred(model_output, t, sample, generator=generator)

            sample, sample_mean = output.prev_sample, output.prev_sample_mean

        sample = sample_mean.clamp(0, 1)
        sample = sample.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            sample = self.numpy_to_pil(sample)

        if not return_dict:
            return (sample,)

        return ImagePipelineOutput(images=sample)

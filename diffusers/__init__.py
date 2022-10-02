from .utils import (
    is_inflect_available,
    is_onnx_available,
    is_scipy_available,
    is_transformers_available,
    is_unidecode_available,
)


__version__ = "0.3.0"

from .configuration_utils import ConfigMixin
from .modeling_utils import ModelMixin
from .models import AutoencoderKL, UNet2DConditionModel, UNet2DModel, VQModel
from .onnx_utils import OnnxRuntimeModel
from .optimization import (
    get_constant_schedule,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
    get_scheduler,
)
from .pipeline_utils import DiffusionPipeline
from .pipelines import DDIMPipeline, DDPMPipeline, KarrasVePipeline, LDMPipeline, PNDMPipeline, ScoreSdeVePipeline
from .schedulers import (
    DDIMScheduler,
    DDPMScheduler,
    KarrasVeScheduler,
    PNDMScheduler,
    SchedulerMixin,
    ScoreSdeVeScheduler,
)
from .utils import logging


if is_scipy_available():
    from .schedulers import LMSDiscreteScheduler
else:
    from .utils.dummy_scipy_objects import *  # noqa F403

from .training_utils import EMAModel


if is_transformers_available():
    from .pipelines import (
        LDMTextToImagePipeline,
        StableDiffusionImg2ImgPipeline,
        StableDiffusionInpaintPipeline,
        StableDiffusionPipeline,
    )
else:
    from .utils.dummy_transformers_objects import *  # noqa F403


if is_transformers_available() and is_onnx_available():
    from .pipelines import StableDiffusionOnnxPipeline
else:
    from .utils.dummy_transformers_and_onnx_objects import *  # noqa F403

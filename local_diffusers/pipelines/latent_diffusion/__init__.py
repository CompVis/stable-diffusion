# flake8: noqa
from ...utils import is_transformers_available


if is_transformers_available():
    from .pipeline_latent_diffusion import LDMBertModel, LDMTextToImagePipeline

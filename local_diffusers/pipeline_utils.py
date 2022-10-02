# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import importlib
import inspect
import os
from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import torch

import diffusers
import PIL
from huggingface_hub import snapshot_download
from PIL import Image
from tqdm.auto import tqdm

from .configuration_utils import ConfigMixin
from .utils import DIFFUSERS_CACHE, BaseOutput, logging


INDEX_FILE = "diffusion_pytorch_model.bin"


logger = logging.get_logger(__name__)


LOADABLE_CLASSES = {
    "diffusers": {
        "ModelMixin": ["save_pretrained", "from_pretrained"],
        "SchedulerMixin": ["save_config", "from_config"],
        "DiffusionPipeline": ["save_pretrained", "from_pretrained"],
        "OnnxRuntimeModel": ["save_pretrained", "from_pretrained"],
    },
    "transformers": {
        "PreTrainedTokenizer": ["save_pretrained", "from_pretrained"],
        "PreTrainedTokenizerFast": ["save_pretrained", "from_pretrained"],
        "PreTrainedModel": ["save_pretrained", "from_pretrained"],
        "FeatureExtractionMixin": ["save_pretrained", "from_pretrained"],
    },
}

ALL_IMPORTABLE_CLASSES = {}
for library in LOADABLE_CLASSES:
    ALL_IMPORTABLE_CLASSES.update(LOADABLE_CLASSES[library])


@dataclass
class ImagePipelineOutput(BaseOutput):
    """
    Output class for image pipelines.

    Args:
        images (`List[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or numpy array of shape `(batch_size, height, width,
            num_channels)`. PIL images or numpy array present the denoised images of the diffusion pipeline.
    """

    images: Union[List[PIL.Image.Image], np.ndarray]


class DiffusionPipeline(ConfigMixin):
    r"""
    Base class for all models.

    [`DiffusionPipeline`] takes care of storing all components (models, schedulers, processors) for diffusion pipelines
    and handles methods for loading, downloading and saving models as well as a few methods common to all pipelines to:

        - move all PyTorch modules to the device of your choice
        - enabling/disabling the progress bar for the denoising iteration

    Class attributes:

        - **config_name** ([`str`]) -- name of the config file that will store the class and module names of all
          compenents of the diffusion pipeline.
    """
    config_name = "model_index.json"

    def register_modules(self, **kwargs):
        # import it here to avoid circular import
        from diffusers import pipelines

        for name, module in kwargs.items():
            # retrive library
            library = module.__module__.split(".")[0]

            # check if the module is a pipeline module
            pipeline_dir = module.__module__.split(".")[-2]
            path = module.__module__.split(".")
            is_pipeline_module = pipeline_dir in path and hasattr(pipelines, pipeline_dir)

            # if library is not in LOADABLE_CLASSES, then it is a custom module.
            # Or if it's a pipeline module, then the module is inside the pipeline
            # folder so we set the library to module name.
            if library not in LOADABLE_CLASSES or is_pipeline_module:
                library = pipeline_dir

            # retrive class_name
            class_name = module.__class__.__name__

            register_dict = {name: (library, class_name)}

            # save model index config
            self.register_to_config(**register_dict)

            # set models
            setattr(self, name, module)

    def save_pretrained(self, save_directory: Union[str, os.PathLike]):
        """
        Save all variables of the pipeline that can be saved and loaded as well as the pipelines configuration file to
        a directory. A pipeline variable can be saved and loaded if its class implements both a save and loading
        method. The pipeline can easily be re-loaded using the `[`~DiffusionPipeline.from_pretrained`]` class method.

        Arguments:
            save_directory (`str` or `os.PathLike`):
                Directory to which to save. Will be created if it doesn't exist.
        """
        self.save_config(save_directory)

        model_index_dict = dict(self.config)
        model_index_dict.pop("_class_name")
        model_index_dict.pop("_diffusers_version")
        model_index_dict.pop("_module", None)

        for pipeline_component_name in model_index_dict.keys():
            sub_model = getattr(self, pipeline_component_name)
            model_cls = sub_model.__class__

            save_method_name = None
            # search for the model's base class in LOADABLE_CLASSES
            for library_name, library_classes in LOADABLE_CLASSES.items():
                library = importlib.import_module(library_name)
                for base_class, save_load_methods in library_classes.items():
                    class_candidate = getattr(library, base_class)
                    if issubclass(model_cls, class_candidate):
                        # if we found a suitable base class in LOADABLE_CLASSES then grab its save method
                        save_method_name = save_load_methods[0]
                        break
                if save_method_name is not None:
                    break

            save_method = getattr(sub_model, save_method_name)
            save_method(os.path.join(save_directory, pipeline_component_name))

    def to(self, torch_device: Optional[Union[str, torch.device]] = None):
        if torch_device is None:
            return self

        module_names, _ = self.extract_init_dict(dict(self.config))
        for name in module_names.keys():
            module = getattr(self, name)
            if isinstance(module, torch.nn.Module):
                module.to(torch_device)
        return self

    @property
    def device(self) -> torch.device:
        r"""
        Returns:
            `torch.device`: The torch device on which the pipeline is located.
        """
        module_names, _ = self.extract_init_dict(dict(self.config))
        for name in module_names.keys():
            module = getattr(self, name)
            if isinstance(module, torch.nn.Module):
                return module.device
        return torch.device("cpu")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], **kwargs):
        r"""
        Instantiate a PyTorch diffusion pipeline from pre-trained pipeline weights.

        The pipeline is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated).

        The warning *Weights from XXX not initialized from pretrained model* means that the weights of XXX do not come
        pretrained with the rest of the model. It is up to you to train those weights with a downstream fine-tuning
        task.

        The warning *Weights from XXX not used in YYY* means that the layer XXX is not used by YYY, therefore those
        weights are discarded.

        Parameters:
            pretrained_model_name_or_path (`str` or `os.PathLike`, *optional*):
                Can be either:

                    - A string, the *repo id* of a pretrained pipeline hosted inside a model repo on
                      https://huggingface.co/ Valid repo ids have to be located under a user or organization name, like
                      `CompVis/ldm-text2im-large-256`.
                    - A path to a *directory* containing pipeline weights saved using
                      [`~DiffusionPipeline.save_pretrained`], e.g., `./my_pipeline_directory/`.
            torch_dtype (`str` or `torch.dtype`, *optional*):
                Override the default `torch.dtype` and load the model under this dtype. If `"auto"` is passed the dtype
                will be automatically derived from the model's weights.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            resume_download (`bool`, *optional*, defaults to `False`):
                Whether or not to delete incompletely received files. Will attempt to resume the download if such a
                file exists.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            output_loading_info(`bool`, *optional*, defaults to `False`):
                Whether ot not to also return a dictionary containing missing keys, unexpected keys and error messages.
            local_files_only(`bool`, *optional*, defaults to `False`):
                Whether or not to only look at local files (i.e., do not try to download the model).
            use_auth_token (`str` or *bool*, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
                when running `huggingface-cli login` (stored in `~/.huggingface`).
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
                identifier allowed by git.
            mirror (`str`, *optional*):
                Mirror source to accelerate downloads in China. If you are from China and have an accessibility
                problem, you can set this option to resolve it. Note that we do not guarantee the timeliness or safety.
                Please refer to the mirror site for more information. specify the folder name here.

            kwargs (remaining dictionary of keyword arguments, *optional*):
                Can be used to overwrite load - and saveable variables - *i.e.* the pipeline components - of the
                speficic pipeline class. The overritten components are then directly passed to the pipelines `__init__`
                method. See example below for more information.

        <Tip>

        Passing `use_auth_token=True`` is required when you want to use a private model, *e.g.*
        `"CompVis/stable-diffusion-v1-4"`

        </Tip>

        <Tip>

        Activate the special ["offline-mode"](https://huggingface.co/diffusers/installation.html#offline-mode) to use
        this method in a firewalled environment.

        </Tip>

        Examples:

        ```py
        >>> from diffusers import DiffusionPipeline

        >>> # Download pipeline from huggingface.co and cache.
        >>> pipeline = DiffusionPipeline.from_pretrained("CompVis/ldm-text2im-large-256")

        >>> # Download pipeline that requires an authorization token
        >>> # For more information on access tokens, please refer to this section
        >>> # of the documentation](https://huggingface.co/docs/hub/security-tokens)
        >>> pipeline = DiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=True)

        >>> # Download pipeline, but overwrite scheduler
        >>> from diffusers import LMSDiscreteScheduler

        >>> scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear")
        >>> pipeline = DiffusionPipeline.from_pretrained(
        ...     "CompVis/stable-diffusion-v1-4", scheduler=scheduler, use_auth_token=True
        ... )
        ```
        """
        cache_dir = kwargs.pop("cache_dir", DIFFUSERS_CACHE)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", False)
        use_auth_token = kwargs.pop("use_auth_token", None)
        revision = kwargs.pop("revision", None)
        torch_dtype = kwargs.pop("torch_dtype", None)
        provider = kwargs.pop("provider", None)

        # 1. Download the checkpoints and configs
        # use snapshot download here to get it working from from_pretrained
        if not os.path.isdir(pretrained_model_name_or_path):
            cached_folder = snapshot_download(
                pretrained_model_name_or_path,
                cache_dir=cache_dir,
                resume_download=resume_download,
                proxies=proxies,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token,
                revision=revision,
            )
        else:
            cached_folder = pretrained_model_name_or_path

        config_dict = cls.get_config_dict(cached_folder)

        # 2. Load the pipeline class, if using custom module then load it from the hub
        # if we load from explicit class, let's use it
        if cls != DiffusionPipeline:
            pipeline_class = cls
        else:
            diffusers_module = importlib.import_module(cls.__module__.split(".")[0])
            pipeline_class = getattr(diffusers_module, config_dict["_class_name"])

        # some modules can be passed directly to the init
        # in this case they are already instantiated in `kwargs`
        # extract them here
        expected_modules = set(inspect.signature(pipeline_class.__init__).parameters.keys())
        passed_class_obj = {k: kwargs.pop(k) for k in expected_modules if k in kwargs}

        init_dict, _ = pipeline_class.extract_init_dict(config_dict, **kwargs)

        init_kwargs = {}

        # import it here to avoid circular import
        from diffusers import pipelines

        # 3. Load each module in the pipeline
        for name, (library_name, class_name) in init_dict.items():
            is_pipeline_module = hasattr(pipelines, library_name)
            loaded_sub_model = None

            # if the model is in a pipeline module, then we load it from the pipeline
            if name in passed_class_obj:
                # 1. check that passed_class_obj has correct parent class
                if not is_pipeline_module:
                    library = importlib.import_module(library_name)
                    class_obj = getattr(library, class_name)
                    importable_classes = LOADABLE_CLASSES[library_name]
                    class_candidates = {c: getattr(library, c) for c in importable_classes.keys()}

                    expected_class_obj = None
                    for class_name, class_candidate in class_candidates.items():
                        if issubclass(class_obj, class_candidate):
                            expected_class_obj = class_candidate

                    if not issubclass(passed_class_obj[name].__class__, expected_class_obj):
                        raise ValueError(
                            f"{passed_class_obj[name]} is of type: {type(passed_class_obj[name])}, but should be"
                            f" {expected_class_obj}"
                        )
                else:
                    logger.warn(
                        f"You have passed a non-standard module {passed_class_obj[name]}. We cannot verify whether it"
                        " has the correct type"
                    )

                # set passed class object
                loaded_sub_model = passed_class_obj[name]
            elif is_pipeline_module:
                pipeline_module = getattr(pipelines, library_name)
                class_obj = getattr(pipeline_module, class_name)
                importable_classes = ALL_IMPORTABLE_CLASSES
                class_candidates = {c: class_obj for c in importable_classes.keys()}
            else:
                # else we just import it from the library.
                library = importlib.import_module(library_name)
                class_obj = getattr(library, class_name)
                importable_classes = LOADABLE_CLASSES[library_name]
                class_candidates = {c: getattr(library, c) for c in importable_classes.keys()}

            if loaded_sub_model is None:
                load_method_name = None
                for class_name, class_candidate in class_candidates.items():
                    if issubclass(class_obj, class_candidate):
                        load_method_name = importable_classes[class_name][1]

                load_method = getattr(class_obj, load_method_name)

                loading_kwargs = {}
                if issubclass(class_obj, torch.nn.Module):
                    loading_kwargs["torch_dtype"] = torch_dtype
                if issubclass(class_obj, diffusers.OnnxRuntimeModel):
                    loading_kwargs["provider"] = provider

                # check if the module is in a subdirectory
                if os.path.isdir(os.path.join(cached_folder, name)):
                    loaded_sub_model = load_method(os.path.join(cached_folder, name), **loading_kwargs)
                else:
                    # else load from the root directory
                    loaded_sub_model = load_method(cached_folder, **loading_kwargs)

            init_kwargs[name] = loaded_sub_model  # UNet(...), # DiffusionSchedule(...)

        # 4. Instantiate the pipeline
        model = pipeline_class(**init_kwargs)
        return model

    @staticmethod
    def numpy_to_pil(images):
        """
        Convert a numpy image or a batch of images to a PIL image.
        """
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]

        return pil_images

    def progress_bar(self, iterable):
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        elif not isinstance(self._progress_bar_config, dict):
            raise ValueError(
                f"`self._progress_bar_config` should be of type `dict`, but is {type(self._progress_bar_config)}."
            )

        return tqdm(iterable, **self._progress_bar_config)

    def set_progress_bar_config(self, **kwargs):
        self._progress_bar_config = kwargs

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

import os
from typing import Callable, List, Optional, Tuple, Union

import torch
from torch import Tensor, device

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError, RevisionNotFoundError
from requests import HTTPError

from .utils import CONFIG_NAME, DIFFUSERS_CACHE, HUGGINGFACE_CO_RESOLVE_ENDPOINT, logging


WEIGHTS_NAME = "diffusion_pytorch_model.bin"


logger = logging.get_logger(__name__)


def get_parameter_device(parameter: torch.nn.Module):
    try:
        return next(parameter.parameters()).device
    except StopIteration:
        # For torch.nn.DataParallel compatibility in PyTorch 1.5

        def find_tensor_attributes(module: torch.nn.Module) -> List[Tuple[str, Tensor]]:
            tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
            return tuples

        gen = parameter._named_members(get_members_fn=find_tensor_attributes)
        first_tuple = next(gen)
        return first_tuple[1].device


def get_parameter_dtype(parameter: torch.nn.Module):
    try:
        return next(parameter.parameters()).dtype
    except StopIteration:
        # For torch.nn.DataParallel compatibility in PyTorch 1.5

        def find_tensor_attributes(module: torch.nn.Module) -> List[Tuple[str, Tensor]]:
            tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
            return tuples

        gen = parameter._named_members(get_members_fn=find_tensor_attributes)
        first_tuple = next(gen)
        return first_tuple[1].dtype


def load_state_dict(checkpoint_file: Union[str, os.PathLike]):
    """
    Reads a PyTorch checkpoint file, returning properly formatted errors if they arise.
    """
    try:
        return torch.load(checkpoint_file, map_location="cpu")
    except Exception as e:
        try:
            with open(checkpoint_file) as f:
                if f.read().startswith("version"):
                    raise OSError(
                        "You seem to have cloned a repository without having git-lfs installed. Please install "
                        "git-lfs and run `git lfs install` followed by `git lfs pull` in the folder "
                        "you cloned."
                    )
                else:
                    raise ValueError(
                        f"Unable to locate the file {checkpoint_file} which is necessary to load this pretrained "
                        "model. Make sure you have saved the model properly."
                    ) from e
        except (UnicodeDecodeError, ValueError):
            raise OSError(
                f"Unable to load weights from pytorch checkpoint file for '{checkpoint_file}' "
                f"at '{checkpoint_file}'. "
                "If you tried to load a PyTorch model from a TF 2.0 checkpoint, please set from_tf=True."
            )


def _load_state_dict_into_model(model_to_load, state_dict):
    # Convert old format to new format if needed from a PyTorch state_dict
    # copy state_dict so _load_from_state_dict can modify it
    state_dict = state_dict.copy()
    error_msgs = []

    # PyTorch's `_load_from_state_dict` does not copy parameters in a module's descendants
    # so we need to apply the function recursively.
    def load(module: torch.nn.Module, prefix=""):
        args = (state_dict, prefix, {}, True, [], [], error_msgs)
        module._load_from_state_dict(*args)

        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + ".")

    load(model_to_load)

    return error_msgs


class ModelMixin(torch.nn.Module):
    r"""
    Base class for all models.

    [`ModelMixin`] takes care of storing the configuration of the models and handles methods for loading, downloading
    and saving models.

        - **config_name** ([`str`]) -- A filename under which the model should be stored when calling
          [`~modeling_utils.ModelMixin.save_pretrained`].
    """
    config_name = CONFIG_NAME
    _automatically_saved_args = ["_diffusers_version", "_class_name", "_name_or_path"]

    def __init__(self):
        super().__init__()

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        is_main_process: bool = True,
        save_function: Callable = torch.save,
    ):
        """
        Save a model and its configuration file to a directory, so that it can be re-loaded using the
        `[`~modeling_utils.ModelMixin.from_pretrained`]` class method.

        Arguments:
            save_directory (`str` or `os.PathLike`):
                Directory to which to save. Will be created if it doesn't exist.
            is_main_process (`bool`, *optional*, defaults to `True`):
                Whether the process calling this is the main process or not. Useful when in distributed training like
                TPUs and need to call this function on all processes. In this case, set `is_main_process=True` only on
                the main process to avoid race conditions.
            save_function (`Callable`):
                The function to use to save the state dictionary. Useful on distributed training like TPUs when one
                need to replace `torch.save` by another method.
        """
        if os.path.isfile(save_directory):
            logger.error(f"Provided path ({save_directory}) should be a directory, not a file")
            return

        os.makedirs(save_directory, exist_ok=True)

        model_to_save = self

        # Attach architecture to the config
        # Save the config
        if is_main_process:
            model_to_save.save_config(save_directory)

        # Save the model
        state_dict = model_to_save.state_dict()

        # Clean the folder from a previous save
        for filename in os.listdir(save_directory):
            full_filename = os.path.join(save_directory, filename)
            # If we have a shard file that is not going to be replaced, we delete it, but only from the main process
            # in distributed settings to avoid race conditions.
            if filename.startswith(WEIGHTS_NAME[:-4]) and os.path.isfile(full_filename) and is_main_process:
                os.remove(full_filename)

        # Save the model
        save_function(state_dict, os.path.join(save_directory, WEIGHTS_NAME))

        logger.info(f"Model weights saved in {os.path.join(save_directory, WEIGHTS_NAME)}")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], **kwargs):
        r"""
        Instantiate a pretrained pytorch model from a pre-trained model configuration.

        The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated). To train
        the model, you should first set it back in training mode with `model.train()`.

        The warning *Weights from XXX not initialized from pretrained model* means that the weights of XXX do not come
        pretrained with the rest of the model. It is up to you to train those weights with a downstream fine-tuning
        task.

        The warning *Weights from XXX not used in YYY* means that the layer XXX is not used by YYY, therefore those
        weights are discarded.

        Parameters:
            pretrained_model_name_or_path (`str` or `os.PathLike`, *optional*):
                Can be either:

                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
                      Valid model ids should have an organization name, like `google/ddpm-celebahq-256`.
                    - A path to a *directory* containing model weights saved using [`~ModelMixin.save_config`], e.g.,
                      `./my_model_directory/`.

            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
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
                when running `diffusers-cli login` (stored in `~/.huggingface`).
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
                identifier allowed by git.
            mirror (`str`, *optional*):
                Mirror source to accelerate downloads in China. If you are from China and have an accessibility
                problem, you can set this option to resolve it. Note that we do not guarantee the timeliness or safety.
                Please refer to the mirror site for more information.

        <Tip>

        Passing `use_auth_token=True`` is required when you want to use a private model.

        </Tip>

        <Tip>

        Activate the special ["offline-mode"](https://huggingface.co/diffusers/installation.html#offline-mode) to use
        this method in a firewalled environment.

        </Tip>

        """
        cache_dir = kwargs.pop("cache_dir", DIFFUSERS_CACHE)
        ignore_mismatched_sizes = kwargs.pop("ignore_mismatched_sizes", False)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        output_loading_info = kwargs.pop("output_loading_info", False)
        local_files_only = kwargs.pop("local_files_only", False)
        use_auth_token = kwargs.pop("use_auth_token", None)
        revision = kwargs.pop("revision", None)
        from_auto_class = kwargs.pop("_from_auto", False)
        torch_dtype = kwargs.pop("torch_dtype", None)
        subfolder = kwargs.pop("subfolder", None)

        user_agent = {"file_type": "model", "framework": "pytorch", "from_auto_class": from_auto_class}

        # Load config if we don't provide a configuration
        config_path = pretrained_model_name_or_path
        model, unused_kwargs = cls.from_config(
            config_path,
            cache_dir=cache_dir,
            return_unused_kwargs=True,
            force_download=force_download,
            resume_download=resume_download,
            proxies=proxies,
            local_files_only=local_files_only,
            use_auth_token=use_auth_token,
            revision=revision,
            subfolder=subfolder,
            **kwargs,
        )

        if torch_dtype is not None and not isinstance(torch_dtype, torch.dtype):
            raise ValueError(
                f"{torch_dtype} needs to be of type `torch.dtype`, e.g. `torch.float16`, but is {type(torch_dtype)}."
            )
        elif torch_dtype is not None:
            model = model.to(torch_dtype)

        model.register_to_config(_name_or_path=pretrained_model_name_or_path)
        # This variable will flag if we're loading a sharded checkpoint. In this case the archive file is just the
        # Load model
        pretrained_model_name_or_path = str(pretrained_model_name_or_path)
        if os.path.isdir(pretrained_model_name_or_path):
            if os.path.isfile(os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)):
                # Load from a PyTorch checkpoint
                model_file = os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)
            elif subfolder is not None and os.path.isfile(
                os.path.join(pretrained_model_name_or_path, subfolder, WEIGHTS_NAME)
            ):
                model_file = os.path.join(pretrained_model_name_or_path, subfolder, WEIGHTS_NAME)
            else:
                raise EnvironmentError(
                    f"Error no file named {WEIGHTS_NAME} found in directory {pretrained_model_name_or_path}."
                )
        else:
            try:
                # Load from URL or cache if already cached
                model_file = hf_hub_download(
                    pretrained_model_name_or_path,
                    filename=WEIGHTS_NAME,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    local_files_only=local_files_only,
                    use_auth_token=use_auth_token,
                    user_agent=user_agent,
                    subfolder=subfolder,
                    revision=revision,
                )

            except RepositoryNotFoundError:
                raise EnvironmentError(
                    f"{pretrained_model_name_or_path} is not a local folder and is not a valid model identifier "
                    "listed on 'https://huggingface.co/models'\nIf this is a private repository, make sure to pass a "
                    "token having permission to this repo with `use_auth_token` or log in with `huggingface-cli "
                    "login` and pass `use_auth_token=True`."
                )
            except RevisionNotFoundError:
                raise EnvironmentError(
                    f"{revision} is not a valid git identifier (branch name, tag name or commit id) that exists for "
                    "this model name. Check the model page at "
                    f"'https://huggingface.co/{pretrained_model_name_or_path}' for available revisions."
                )
            except EntryNotFoundError:
                raise EnvironmentError(
                    f"{pretrained_model_name_or_path} does not appear to have a file named {WEIGHTS_NAME}."
                )
            except HTTPError as err:
                raise EnvironmentError(
                    "There was a specific connection error when trying to load"
                    f" {pretrained_model_name_or_path}:\n{err}"
                )
            except ValueError:
                raise EnvironmentError(
                    f"We couldn't connect to '{HUGGINGFACE_CO_RESOLVE_ENDPOINT}' to load this model, couldn't find it"
                    f" in the cached files and it looks like {pretrained_model_name_or_path} is not the path to a"
                    f" directory containing a file named {WEIGHTS_NAME} or"
                    " \nCheckout your internet connection or see how to run the library in"
                    " offline mode at 'https://huggingface.co/docs/diffusers/installation#offline-mode'."
                )
            except EnvironmentError:
                raise EnvironmentError(
                    f"Can't load the model for '{pretrained_model_name_or_path}'. If you were trying to load it from "
                    "'https://huggingface.co/models', make sure you don't have a local directory with the same name. "
                    f"Otherwise, make sure '{pretrained_model_name_or_path}' is the correct path to a directory "
                    f"containing a file named {WEIGHTS_NAME}"
                )

            # restore default dtype
        state_dict = load_state_dict(model_file)
        model, missing_keys, unexpected_keys, mismatched_keys, error_msgs = cls._load_pretrained_model(
            model,
            state_dict,
            model_file,
            pretrained_model_name_or_path,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
        )

        # Set model in evaluation mode to deactivate DropOut modules by default
        model.eval()

        if output_loading_info:
            loading_info = {
                "missing_keys": missing_keys,
                "unexpected_keys": unexpected_keys,
                "mismatched_keys": mismatched_keys,
                "error_msgs": error_msgs,
            }
            return model, loading_info

        return model

    @classmethod
    def _load_pretrained_model(
        cls,
        model,
        state_dict,
        resolved_archive_file,
        pretrained_model_name_or_path,
        ignore_mismatched_sizes=False,
    ):
        # Retrieve missing & unexpected_keys
        model_state_dict = model.state_dict()
        loaded_keys = [k for k in state_dict.keys()]

        expected_keys = list(model_state_dict.keys())

        original_loaded_keys = loaded_keys

        missing_keys = list(set(expected_keys) - set(loaded_keys))
        unexpected_keys = list(set(loaded_keys) - set(expected_keys))

        # Make sure we are able to load base models as well as derived models (with heads)
        model_to_load = model

        def _find_mismatched_keys(
            state_dict,
            model_state_dict,
            loaded_keys,
            ignore_mismatched_sizes,
        ):
            mismatched_keys = []
            if ignore_mismatched_sizes:
                for checkpoint_key in loaded_keys:
                    model_key = checkpoint_key

                    if (
                        model_key in model_state_dict
                        and state_dict[checkpoint_key].shape != model_state_dict[model_key].shape
                    ):
                        mismatched_keys.append(
                            (checkpoint_key, state_dict[checkpoint_key].shape, model_state_dict[model_key].shape)
                        )
                        del state_dict[checkpoint_key]
            return mismatched_keys

        if state_dict is not None:
            # Whole checkpoint
            mismatched_keys = _find_mismatched_keys(
                state_dict,
                model_state_dict,
                original_loaded_keys,
                ignore_mismatched_sizes,
            )
            error_msgs = _load_state_dict_into_model(model_to_load, state_dict)

        if len(error_msgs) > 0:
            error_msg = "\n\t".join(error_msgs)
            if "size mismatch" in error_msg:
                error_msg += (
                    "\n\tYou may consider adding `ignore_mismatched_sizes=True` in the model `from_pretrained` method."
                )
            raise RuntimeError(f"Error(s) in loading state_dict for {model.__class__.__name__}:\n\t{error_msg}")

        if len(unexpected_keys) > 0:
            logger.warning(
                f"Some weights of the model checkpoint at {pretrained_model_name_or_path} were not used when"
                f" initializing {model.__class__.__name__}: {unexpected_keys}\n- This IS expected if you are"
                f" initializing {model.__class__.__name__} from the checkpoint of a model trained on another task"
                " or with another architecture (e.g. initializing a BertForSequenceClassification model from a"
                " BertForPreTraining model).\n- This IS NOT expected if you are initializing"
                f" {model.__class__.__name__} from the checkpoint of a model that you expect to be exactly"
                " identical (initializing a BertForSequenceClassification model from a"
                " BertForSequenceClassification model)."
            )
        else:
            logger.info(f"All model checkpoint weights were used when initializing {model.__class__.__name__}.\n")
        if len(missing_keys) > 0:
            logger.warning(
                f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at"
                f" {pretrained_model_name_or_path} and are newly initialized: {missing_keys}\nYou should probably"
                " TRAIN this model on a down-stream task to be able to use it for predictions and inference."
            )
        elif len(mismatched_keys) == 0:
            logger.info(
                f"All the weights of {model.__class__.__name__} were initialized from the model checkpoint at"
                f" {pretrained_model_name_or_path}.\nIf your task is similar to the task the model of the"
                f" checkpoint was trained on, you can already use {model.__class__.__name__} for predictions"
                " without further training."
            )
        if len(mismatched_keys) > 0:
            mismatched_warning = "\n".join(
                [
                    f"- {key}: found shape {shape1} in the checkpoint and {shape2} in the model instantiated"
                    for key, shape1, shape2 in mismatched_keys
                ]
            )
            logger.warning(
                f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at"
                f" {pretrained_model_name_or_path} and are newly initialized because the shapes did not"
                f" match:\n{mismatched_warning}\nYou should probably TRAIN this model on a down-stream task to be"
                " able to use it for predictions and inference."
            )

        return model, missing_keys, unexpected_keys, mismatched_keys, error_msgs

    @property
    def device(self) -> device:
        """
        `torch.device`: The device on which the module is (assuming that all the module parameters are on the same
        device).
        """
        return get_parameter_device(self)

    @property
    def dtype(self) -> torch.dtype:
        """
        `torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        return get_parameter_dtype(self)

    def num_parameters(self, only_trainable: bool = False, exclude_embeddings: bool = False) -> int:
        """
        Get number of (optionally, trainable or non-embeddings) parameters in the module.

        Args:
            only_trainable (`bool`, *optional*, defaults to `False`):
                Whether or not to return only the number of trainable parameters

            exclude_embeddings (`bool`, *optional*, defaults to `False`):
                Whether or not to return only the number of non-embeddings parameters

        Returns:
            `int`: The number of parameters.
        """

        if exclude_embeddings:
            embedding_param_names = [
                f"{name}.weight"
                for name, module_type in self.named_modules()
                if isinstance(module_type, torch.nn.Embedding)
            ]
            non_embedding_parameters = [
                parameter for name, parameter in self.named_parameters() if name not in embedding_param_names
            ]
            return sum(p.numel() for p in non_embedding_parameters if p.requires_grad or not only_trainable)
        else:
            return sum(p.numel() for p in self.parameters() if p.requires_grad or not only_trainable)


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """
    Recursively unwraps a model from potential containers (as used in distributed training).

    Args:
        model (`torch.nn.Module`): The model to unwrap.
    """
    # since there could be multiple levels of wrapping, unwrap recursively
    if hasattr(model, "module"):
        return unwrap_model(model.module)
    else:
        return model

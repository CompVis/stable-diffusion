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
""" ConfigMixinuration base class and utilities."""
import functools
import inspect
import json
import os
import re
from collections import OrderedDict
from typing import Any, Dict, Tuple, Union

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError, RevisionNotFoundError
from requests import HTTPError

from . import __version__
from .utils import DIFFUSERS_CACHE, HUGGINGFACE_CO_RESOLVE_ENDPOINT, logging


logger = logging.get_logger(__name__)

_re_configuration_file = re.compile(r"config\.(.*)\.json")


class ConfigMixin:
    r"""
    Base class for all configuration classes. Stores all configuration parameters under `self.config` Also handles all
    methods for loading/downloading/saving classes inheriting from [`ConfigMixin`] with
        - [`~ConfigMixin.from_config`]
        - [`~ConfigMixin.save_config`]

    Class attributes:
        - **config_name** (`str`) -- A filename under which the config should stored when calling
          [`~ConfigMixin.save_config`] (should be overriden by parent class).
        - **ignore_for_config** (`List[str]`) -- A list of attributes that should not be saved in the config (should be
          overriden by parent class).
    """
    config_name = None
    ignore_for_config = []

    def register_to_config(self, **kwargs):
        if self.config_name is None:
            raise NotImplementedError(f"Make sure that {self.__class__} has defined a class name `config_name`")
        kwargs["_class_name"] = self.__class__.__name__
        kwargs["_diffusers_version"] = __version__

        for key, value in kwargs.items():
            try:
                setattr(self, key, value)
            except AttributeError as err:
                logger.error(f"Can't set {key} with value {value} for {self}")
                raise err

        if not hasattr(self, "_internal_dict"):
            internal_dict = kwargs
        else:
            previous_dict = dict(self._internal_dict)
            internal_dict = {**self._internal_dict, **kwargs}
            logger.debug(f"Updating config from {previous_dict} to {internal_dict}")

        self._internal_dict = FrozenDict(internal_dict)

    def save_config(self, save_directory: Union[str, os.PathLike], push_to_hub: bool = False, **kwargs):
        """
        Save a configuration object to the directory `save_directory`, so that it can be re-loaded using the
        [`~ConfigMixin.from_config`] class method.

        Args:
            save_directory (`str` or `os.PathLike`):
                Directory where the configuration JSON file will be saved (will be created if it does not exist).
        """
        if os.path.isfile(save_directory):
            raise AssertionError(f"Provided path ({save_directory}) should be a directory, not a file")

        os.makedirs(save_directory, exist_ok=True)

        # If we save using the predefined names, we can load using `from_config`
        output_config_file = os.path.join(save_directory, self.config_name)

        self.to_json_file(output_config_file)
        logger.info(f"ConfigMixinuration saved in {output_config_file}")

    @classmethod
    def from_config(cls, pretrained_model_name_or_path: Union[str, os.PathLike], return_unused_kwargs=False, **kwargs):
        r"""
        Instantiate a Python class from a pre-defined JSON-file.

        Parameters:
            pretrained_model_name_or_path (`str` or `os.PathLike`, *optional*):
                Can be either:

                    - A string, the *model id* of a model repo on huggingface.co. Valid model ids should have an
                      organization name, like `google/ddpm-celebahq-256`.
                    - A path to a *directory* containing model weights saved using [`~ConfigMixin.save_config`], e.g.,
                      `./my_model_directory/`.

            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            ignore_mismatched_sizes (`bool`, *optional*, defaults to `False`):
                Whether or not to raise an error if some of the weights from the checkpoint do not have the same size
                as the weights of the model (if for instance, you are instantiating a model with 10 labels from a
                checkpoint with 3 labels).
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
                when running `transformers-cli login` (stored in `~/.huggingface`).
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

        Activate the special ["offline-mode"](https://huggingface.co/transformers/installation.html#offline-mode) to
        use this method in a firewalled environment.

        </Tip>

        """
        config_dict = cls.get_config_dict(pretrained_model_name_or_path=pretrained_model_name_or_path, **kwargs)

        init_dict, unused_kwargs = cls.extract_init_dict(config_dict, **kwargs)

        model = cls(**init_dict)

        if return_unused_kwargs:
            return model, unused_kwargs
        else:
            return model

    @classmethod
    def get_config_dict(
        cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        cache_dir = kwargs.pop("cache_dir", DIFFUSERS_CACHE)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        use_auth_token = kwargs.pop("use_auth_token", None)
        local_files_only = kwargs.pop("local_files_only", False)
        revision = kwargs.pop("revision", None)
        subfolder = kwargs.pop("subfolder", None)

        user_agent = {"file_type": "config"}

        pretrained_model_name_or_path = str(pretrained_model_name_or_path)

        if cls.config_name is None:
            raise ValueError(
                "`self.config_name` is not defined. Note that one should not load a config from "
                "`ConfigMixin`. Please make sure to define `config_name` in a class inheriting from `ConfigMixin`"
            )

        if os.path.isfile(pretrained_model_name_or_path):
            config_file = pretrained_model_name_or_path
        elif os.path.isdir(pretrained_model_name_or_path):
            if os.path.isfile(os.path.join(pretrained_model_name_or_path, cls.config_name)):
                # Load from a PyTorch checkpoint
                config_file = os.path.join(pretrained_model_name_or_path, cls.config_name)
            elif subfolder is not None and os.path.isfile(
                os.path.join(pretrained_model_name_or_path, subfolder, cls.config_name)
            ):
                config_file = os.path.join(pretrained_model_name_or_path, subfolder, cls.config_name)
            else:
                raise EnvironmentError(
                    f"Error no file named {cls.config_name} found in directory {pretrained_model_name_or_path}."
                )
        else:
            try:
                # Load from URL or cache if already cached
                config_file = hf_hub_download(
                    pretrained_model_name_or_path,
                    filename=cls.config_name,
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
                    f"{pretrained_model_name_or_path} is not a local folder and is not a valid model identifier"
                    " listed on 'https://huggingface.co/models'\nIf this is a private repository, make sure to pass a"
                    " token having permission to this repo with `use_auth_token` or log in with `huggingface-cli"
                    " login` and pass `use_auth_token=True`."
                )
            except RevisionNotFoundError:
                raise EnvironmentError(
                    f"{revision} is not a valid git identifier (branch name, tag name or commit id) that exists for"
                    " this model name. Check the model page at"
                    f" 'https://huggingface.co/{pretrained_model_name_or_path}' for available revisions."
                )
            except EntryNotFoundError:
                raise EnvironmentError(
                    f"{pretrained_model_name_or_path} does not appear to have a file named {cls.config_name}."
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
                    f" directory containing a {cls.config_name} file.\nCheckout your internet connection or see how to"
                    " run the library in offline mode at"
                    " 'https://huggingface.co/docs/diffusers/installation#offline-mode'."
                )
            except EnvironmentError:
                raise EnvironmentError(
                    f"Can't load config for '{pretrained_model_name_or_path}'. If you were trying to load it from "
                    "'https://huggingface.co/models', make sure you don't have a local directory with the same name. "
                    f"Otherwise, make sure '{pretrained_model_name_or_path}' is the correct path to a directory "
                    f"containing a {cls.config_name} file"
                )

        try:
            # Load config dict
            config_dict = cls._dict_from_json_file(config_file)
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise EnvironmentError(f"It looks like the config file at '{config_file}' is not a valid JSON file.")

        return config_dict

    @classmethod
    def extract_init_dict(cls, config_dict, **kwargs):
        expected_keys = set(dict(inspect.signature(cls.__init__).parameters).keys())
        expected_keys.remove("self")
        # remove general kwargs if present in dict
        if "kwargs" in expected_keys:
            expected_keys.remove("kwargs")
        # remove keys to be ignored
        if len(cls.ignore_for_config) > 0:
            expected_keys = expected_keys - set(cls.ignore_for_config)
        init_dict = {}
        for key in expected_keys:
            if key in kwargs:
                # overwrite key
                init_dict[key] = kwargs.pop(key)
            elif key in config_dict:
                # use value from config dict
                init_dict[key] = config_dict.pop(key)

        unused_kwargs = config_dict.update(kwargs)

        passed_keys = set(init_dict.keys())
        if len(expected_keys - passed_keys) > 0:
            logger.warning(
                f"{expected_keys - passed_keys} was not found in config. Values will be initialized to default values."
            )

        return init_dict, unused_kwargs

    @classmethod
    def _dict_from_json_file(cls, json_file: Union[str, os.PathLike]):
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        return json.loads(text)

    def __repr__(self):
        return f"{self.__class__.__name__} {self.to_json_string()}"

    @property
    def config(self) -> Dict[str, Any]:
        return self._internal_dict

    def to_json_string(self) -> str:
        """
        Serializes this instance to a JSON string.

        Returns:
            `str`: String containing all the attributes that make up this configuration instance in JSON format.
        """
        config_dict = self._internal_dict if hasattr(self, "_internal_dict") else {}
        return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path: Union[str, os.PathLike]):
        """
        Save this instance to a JSON file.

        Args:
            json_file_path (`str` or `os.PathLike`):
                Path to the JSON file in which this configuration instance's parameters will be saved.
        """
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string())


class FrozenDict(OrderedDict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        for key, value in self.items():
            setattr(self, key, value)

        self.__frozen = True

    def __delitem__(self, *args, **kwargs):
        raise Exception(f"You cannot use ``__delitem__`` on a {self.__class__.__name__} instance.")

    def setdefault(self, *args, **kwargs):
        raise Exception(f"You cannot use ``setdefault`` on a {self.__class__.__name__} instance.")

    def pop(self, *args, **kwargs):
        raise Exception(f"You cannot use ``pop`` on a {self.__class__.__name__} instance.")

    def update(self, *args, **kwargs):
        raise Exception(f"You cannot use ``update`` on a {self.__class__.__name__} instance.")

    def __setattr__(self, name, value):
        if hasattr(self, "__frozen") and self.__frozen:
            raise Exception(f"You cannot use ``__setattr__`` on a {self.__class__.__name__} instance.")
        super().__setattr__(name, value)

    def __setitem__(self, name, value):
        if hasattr(self, "__frozen") and self.__frozen:
            raise Exception(f"You cannot use ``__setattr__`` on a {self.__class__.__name__} instance.")
        super().__setitem__(name, value)


def register_to_config(init):
    r"""
    Decorator to apply on the init of classes inheriting from [`ConfigMixin`] so that all the arguments are
    automatically sent to `self.register_for_config`. To ignore a specific argument accepted by the init but that
    shouldn't be registered in the config, use the `ignore_for_config` class variable

    Warning: Once decorated, all private arguments (beginning with an underscore) are trashed and not sent to the init!
    """

    @functools.wraps(init)
    def inner_init(self, *args, **kwargs):
        # Ignore private kwargs in the init.
        init_kwargs = {k: v for k, v in kwargs.items() if not k.startswith("_")}
        init(self, *args, **init_kwargs)
        if not isinstance(self, ConfigMixin):
            raise RuntimeError(
                f"`@register_for_config` was applied to {self.__class__.__name__} init method, but this class does "
                "not inherit from `ConfigMixin`."
            )

        ignore = getattr(self, "ignore_for_config", [])
        # Get positional arguments aligned with kwargs
        new_kwargs = {}
        signature = inspect.signature(init)
        parameters = {
            name: p.default for i, (name, p) in enumerate(signature.parameters.items()) if i > 0 and name not in ignore
        }
        for arg, name in zip(args, parameters.keys()):
            new_kwargs[name] = arg

        # Then add all kwargs
        new_kwargs.update(
            {
                k: init_kwargs.get(k, default)
                for k, default in parameters.items()
                if k not in ignore and k not in new_kwargs
            }
        )
        getattr(self, "register_to_config")(**new_kwargs)

    return inner_init

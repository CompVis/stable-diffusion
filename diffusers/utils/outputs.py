# Copyright 2022 The HuggingFace Team. All rights reserved.
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
"""
Generic utilities
"""

import warnings
from collections import OrderedDict
from dataclasses import fields
from typing import Any, Tuple

import numpy as np

from .import_utils import is_torch_available


def is_tensor(x):
    """
    Tests if `x` is a `torch.Tensor` or `np.ndarray`.
    """
    if is_torch_available():
        import torch

        if isinstance(x, torch.Tensor):
            return True

    return isinstance(x, np.ndarray)


class BaseOutput(OrderedDict):
    """
    Base class for all model outputs as dataclass. Has a `__getitem__` that allows indexing by integer or slice (like a
    tuple) or strings (like a dictionary) that will ignore the `None` attributes. Otherwise behaves like a regular
    python dictionary.

    <Tip warning={true}>

    You can't unpack a `BaseOutput` directly. Use the [`~utils.BaseOutput.to_tuple`] method to convert it to a tuple
    before.

    </Tip>
    """

    def __post_init__(self):
        class_fields = fields(self)

        # Safety and consistency checks
        if not len(class_fields):
            raise ValueError(f"{self.__class__.__name__} has no fields.")

        for field in class_fields:
            v = getattr(self, field.name)
            if v is not None:
                self[field.name] = v

    def __delitem__(self, *args, **kwargs):
        raise Exception(f"You cannot use ``__delitem__`` on a {self.__class__.__name__} instance.")

    def setdefault(self, *args, **kwargs):
        raise Exception(f"You cannot use ``setdefault`` on a {self.__class__.__name__} instance.")

    def pop(self, *args, **kwargs):
        raise Exception(f"You cannot use ``pop`` on a {self.__class__.__name__} instance.")

    def update(self, *args, **kwargs):
        raise Exception(f"You cannot use ``update`` on a {self.__class__.__name__} instance.")

    def __getitem__(self, k):
        if isinstance(k, str):
            inner_dict = {k: v for (k, v) in self.items()}
            if self.__class__.__name__ in ["StableDiffusionPipelineOutput", "ImagePipelineOutput"] and k == "sample":
                warnings.warn(
                    "The keyword 'samples' is deprecated and will be removed in version 0.4.0. Please use `.images` or"
                    " `'images'` instead.",
                    DeprecationWarning,
                )
                return inner_dict["images"]
            return inner_dict[k]
        else:
            return self.to_tuple()[k]

    def __setattr__(self, name, value):
        if name in self.keys() and value is not None:
            # Don't call self.__setitem__ to avoid recursion errors
            super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        # Will raise a KeyException if needed
        super().__setitem__(key, value)
        # Don't call self.__setattr__ to avoid recursion errors
        super().__setattr__(key, value)

    def to_tuple(self) -> Tuple[Any]:
        """
        Convert self to a tuple containing all the attributes/keys that are not `None`.
        """
        return tuple(self[k] for k in self.keys())

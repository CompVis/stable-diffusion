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
Import utilities: Utilities related to imports and our lazy inits.
"""
import importlib.util
import os
import sys
from collections import OrderedDict

from packaging import version

from . import logging


# The package importlib_metadata is in a different place, depending on the python version.
if sys.version_info < (3, 8):
    import importlib_metadata
else:
    import importlib.metadata as importlib_metadata


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

ENV_VARS_TRUE_VALUES = {"1", "ON", "YES", "TRUE"}
ENV_VARS_TRUE_AND_AUTO_VALUES = ENV_VARS_TRUE_VALUES.union({"AUTO"})

USE_TF = os.environ.get("USE_TF", "AUTO").upper()
USE_TORCH = os.environ.get("USE_TORCH", "AUTO").upper()
USE_JAX = os.environ.get("USE_FLAX", "AUTO").upper()

_torch_version = "N/A"
if USE_TORCH in ENV_VARS_TRUE_AND_AUTO_VALUES and USE_TF not in ENV_VARS_TRUE_VALUES:
    _torch_available = importlib.util.find_spec("torch") is not None
    if _torch_available:
        try:
            _torch_version = importlib_metadata.version("torch")
            logger.info(f"PyTorch version {_torch_version} available.")
        except importlib_metadata.PackageNotFoundError:
            _torch_available = False
else:
    logger.info("Disabling PyTorch because USE_TF is set")
    _torch_available = False


_tf_version = "N/A"
if USE_TF in ENV_VARS_TRUE_AND_AUTO_VALUES and USE_TORCH not in ENV_VARS_TRUE_VALUES:
    _tf_available = importlib.util.find_spec("tensorflow") is not None
    if _tf_available:
        candidates = (
            "tensorflow",
            "tensorflow-cpu",
            "tensorflow-gpu",
            "tf-nightly",
            "tf-nightly-cpu",
            "tf-nightly-gpu",
            "intel-tensorflow",
            "intel-tensorflow-avx512",
            "tensorflow-rocm",
            "tensorflow-macos",
            "tensorflow-aarch64",
        )
        _tf_version = None
        # For the metadata, we have to look for both tensorflow and tensorflow-cpu
        for pkg in candidates:
            try:
                _tf_version = importlib_metadata.version(pkg)
                break
            except importlib_metadata.PackageNotFoundError:
                pass
        _tf_available = _tf_version is not None
    if _tf_available:
        if version.parse(_tf_version) < version.parse("2"):
            logger.info(f"TensorFlow found but with version {_tf_version}. Diffusers requires version 2 minimum.")
            _tf_available = False
        else:
            logger.info(f"TensorFlow version {_tf_version} available.")
else:
    logger.info("Disabling Tensorflow because USE_TORCH is set")
    _tf_available = False


if USE_JAX in ENV_VARS_TRUE_AND_AUTO_VALUES:
    _flax_available = importlib.util.find_spec("jax") is not None and importlib.util.find_spec("flax") is not None
    if _flax_available:
        try:
            _jax_version = importlib_metadata.version("jax")
            _flax_version = importlib_metadata.version("flax")
            logger.info(f"JAX version {_jax_version}, Flax version {_flax_version} available.")
        except importlib_metadata.PackageNotFoundError:
            _flax_available = False
else:
    _flax_available = False


_transformers_available = importlib.util.find_spec("transformers") is not None
try:
    _transformers_version = importlib_metadata.version("transformers")
    logger.debug(f"Successfully imported transformers version {_transformers_version}")
except importlib_metadata.PackageNotFoundError:
    _transformers_available = False


_inflect_available = importlib.util.find_spec("inflect") is not None
try:
    _inflect_version = importlib_metadata.version("inflect")
    logger.debug(f"Successfully imported inflect version {_inflect_version}")
except importlib_metadata.PackageNotFoundError:
    _inflect_available = False


_unidecode_available = importlib.util.find_spec("unidecode") is not None
try:
    _unidecode_version = importlib_metadata.version("unidecode")
    logger.debug(f"Successfully imported unidecode version {_unidecode_version}")
except importlib_metadata.PackageNotFoundError:
    _unidecode_available = False


_modelcards_available = importlib.util.find_spec("modelcards") is not None
try:
    _modelcards_version = importlib_metadata.version("modelcards")
    logger.debug(f"Successfully imported modelcards version {_modelcards_version}")
except importlib_metadata.PackageNotFoundError:
    _modelcards_available = False


_onnx_available = importlib.util.find_spec("onnxruntime") is not None
try:
    _onnxruntime_version = importlib_metadata.version("onnxruntime")
    logger.debug(f"Successfully imported onnxruntime version {_onnxruntime_version}")
except importlib_metadata.PackageNotFoundError:
    _onnx_available = False


_scipy_available = importlib.util.find_spec("scipy") is not None
try:
    _scipy_version = importlib_metadata.version("scipy")
    logger.debug(f"Successfully imported transformers version {_scipy_version}")
except importlib_metadata.PackageNotFoundError:
    _scipy_available = False


def is_torch_available():
    return _torch_available


def is_tf_available():
    return _tf_available


def is_flax_available():
    return _flax_available


def is_transformers_available():
    return _transformers_available


def is_inflect_available():
    return _inflect_available


def is_unidecode_available():
    return _unidecode_available


def is_modelcards_available():
    return _modelcards_available


def is_onnx_available():
    return _onnx_available


def is_scipy_available():
    return _scipy_available


# docstyle-ignore
FLAX_IMPORT_ERROR = """
{0} requires the FLAX library but it was not found in your environment. Checkout the instructions on the
installation page: https://github.com/google/flax and follow the ones that match your environment.
"""

# docstyle-ignore
INFLECT_IMPORT_ERROR = """
{0} requires the inflect library but it was not found in your environment. You can install it with pip: `pip install
inflect`
"""

# docstyle-ignore
PYTORCH_IMPORT_ERROR = """
{0} requires the PyTorch library but it was not found in your environment. Checkout the instructions on the
installation page: https://pytorch.org/get-started/locally/ and follow the ones that match your environment.
"""

# docstyle-ignore
ONNX_IMPORT_ERROR = """
{0} requires the onnxruntime library but it was not found in your environment. You can install it with pip: `pip
install onnxruntime`
"""

# docstyle-ignore
SCIPY_IMPORT_ERROR = """
{0} requires the scipy library but it was not found in your environment. You can install it with pip: `pip install
scipy`
"""

# docstyle-ignore
TENSORFLOW_IMPORT_ERROR = """
{0} requires the TensorFlow library but it was not found in your environment. Checkout the instructions on the
installation page: https://www.tensorflow.org/install and follow the ones that match your environment.
"""

# docstyle-ignore
TRANSFORMERS_IMPORT_ERROR = """
{0} requires the transformers library but it was not found in your environment. You can install it with pip: `pip
install transformers`
"""

# docstyle-ignore
UNIDECODE_IMPORT_ERROR = """
{0} requires the unidecode library but it was not found in your environment. You can install it with pip: `pip install
Unidecode`
"""


BACKENDS_MAPPING = OrderedDict(
    [
        ("flax", (is_flax_available, FLAX_IMPORT_ERROR)),
        ("inflect", (is_inflect_available, INFLECT_IMPORT_ERROR)),
        ("onnx", (is_onnx_available, ONNX_IMPORT_ERROR)),
        ("scipy", (is_scipy_available, SCIPY_IMPORT_ERROR)),
        ("tf", (is_tf_available, TENSORFLOW_IMPORT_ERROR)),
        ("torch", (is_torch_available, PYTORCH_IMPORT_ERROR)),
        ("transformers", (is_transformers_available, TRANSFORMERS_IMPORT_ERROR)),
        ("unidecode", (is_unidecode_available, UNIDECODE_IMPORT_ERROR)),
    ]
)


def requires_backends(obj, backends):
    if not isinstance(backends, (list, tuple)):
        backends = [backends]

    name = obj.__name__ if hasattr(obj, "__name__") else obj.__class__.__name__
    checks = (BACKENDS_MAPPING[backend] for backend in backends)
    failed = [msg.format(name) for available, msg in checks if not available()]
    if failed:
        raise ImportError("".join(failed))


class DummyObject(type):
    """
    Metaclass for the dummy objects. Any class inheriting from it will return the ImportError generated by
    `requires_backend` each time a user tries to access any method of that class.
    """

    def __getattr__(cls, key):
        if key.startswith("_"):
            return super().__getattr__(cls, key)
        requires_backends(cls, cls._backends)

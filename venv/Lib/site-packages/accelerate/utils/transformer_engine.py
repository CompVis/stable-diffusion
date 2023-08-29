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

import torch.nn as nn

from .imports import is_fp8_available


if is_fp8_available():
    import transformer_engine.pytorch as te


def convert_model(model, to_transformer_engine=True, _convert_linear=True, _convert_ln=True):
    """
    Recursively converts the linear and layernorm layers of a model to their `transformers_engine` counterpart.
    """
    if not is_fp8_available():
        raise ImportError("Using `convert_model` requires transformer_engine to be installed.")
    for name, module in model.named_children():
        if isinstance(module, nn.Linear) and to_transformer_engine and _convert_linear:
            # Return early if the linear layer weights are not multiples of 16
            if any(p % 16 != 0 for p in module.weight.shape):
                return
            has_bias = module.bias is not None
            te_module = te.Linear(
                module.in_features, module.out_features, bias=has_bias, params_dtype=module.weight.dtype
            )
            te_module.weight.data = module.weight.data.clone()
            if has_bias:
                te_module.bias.data = module.bias.data.clone()

            setattr(model, name, te_module)
        elif isinstance(module, nn.LayerNorm) and to_transformer_engine and _convert_ln:
            te_module = te.LayerNorm(module.normalized_shape[0], eps=module.eps, params_dtype=module.weight.dtype)
            te_module.weight.data = module.weight.data.clone()
            te_module.bias.data = module.bias.data.clone()

            setattr(model, name, te_module)
        elif isinstance(module, te.Linear) and not to_transformer_engine and _convert_linear:
            has_bias = module.bias is not None
            new_module = nn.Linear(
                module.in_features, module.out_features, bias=has_bias, params_dtype=module.weight.dtype
            )
            new_module.weight.data = module.weight.data.clone()
            if has_bias:
                new_module.bias.data = module.bias.data.clone()

            setattr(model, name, new_module)
        elif isinstance(module, te.LayerNorm) and not to_transformer_engine and _convert_ln:
            new_module = nn.LayerNorm(module.normalized_shape[0], eps=module.eps, params_dtype=module.weight.dtype)
            new_module.weight.data = module.weight.data.clone()
            new_module.bias.data = module.bias.data.clone()

            setattr(model, name, new_module)
        else:
            convert_model(
                module,
                to_transformer_engine=to_transformer_engine,
                _convert_linear=_convert_linear,
                _convert_ln=_convert_ln,
            )


def has_transformer_engine_layers(model):
    """
    Returns whether a given model has some `transformer_engine` layer or not.
    """
    if not is_fp8_available():
        raise ImportError("Using `has_transformer_engine_layers` requires transformer_engine to be installed.")
    for m in model.modules():
        if isinstance(m, (te.LayerNorm, te.Linear)):
            return True
    return False

from typing import Optional
import torch
from torch import Tensor
from scripts.retro_diffusion import rd
from torch.nn import functional as F
from torch.nn.modules.utils import _pair

def flatten(el):
    # Flatten nested elements by recursively traversing through children
    flattened = [flatten(children) for children in el.children()]
    res = [el]
    for c in flattened:
        res += c
    return res

def patch_conv(**patch):
    # Patch the Conv2d class with a custom __init__ method
    cls = torch.nn.Conv2d
    init = cls.__init__

    def __init__(self, *args, **kwargs):
        # Call the original init method and apply the patch arguments
        return init(self, *args, **kwargs, **patch)

    cls.__init__ = __init__


def patch_conv_asymmetric(model, x, y):
    # Patch Conv2d layers in the given model for asymmetric padding
    for layer in flatten(model):
        if type(layer) == torch.nn.Conv2d:
            # Set padding mode based on x and y arguments
            layer.padding_modeX = "circular" if x else "constant"
            layer.padding_modeY = "circular" if y else "constant"

            # Compute padding values based on reversed padding repeated twice
            layer.paddingX = (
                layer._reversed_padding_repeated_twice[0],
                layer._reversed_padding_repeated_twice[1],
                0,
                0,
            )
            layer.paddingY = (
                0,
                0,
                layer._reversed_padding_repeated_twice[2],
                layer._reversed_padding_repeated_twice[3],
            )

            # Patch the _conv_forward method with a replacement function
            layer._conv_forward = __replacementConv2DConvForward.__get__(
                layer, torch.nn.Conv2d
            )


def restoreConv2DMethods(model):
    # Restore original _conv_forward method for Conv2d layers in the model
    for layer in flatten(model):
        if type(layer) == torch.nn.Conv2d:
            layer._conv_forward = torch.nn.Conv2d._conv_forward.__get__(
                layer, torch.nn.Conv2d
            )


def __replacementConv2DConvForward(
    self, input: Tensor, weight: Tensor, bias: Optional[Tensor]
):
    # Replacement function for Conv2d's _conv_forward method
    working = F.pad(input, self.paddingX, mode=self.padding_modeX)
    working = F.pad(working, self.paddingY, mode=self.padding_modeY)
    return F.conv2d(
        working, weight, bias, self.stride, _pair(0), self.dilation, self.groups
    )


def patch_tiling(tilingX, tilingY, model, modelFS, modelPV):
    # Convert tilingX and tilingY to boolean values
    X = bool(tilingX == "true")
    Y = bool(tilingY == "true")

    # Patch Conv2d layers in the given models for asymmetric padding
    patch_conv_asymmetric(model, X, Y)
    patch_conv_asymmetric(modelFS, X, Y)
    patch_conv_asymmetric(modelPV.model, X, Y)

    if X or Y:
        # Print a message indicating the direction(s) patched for tiling
        rd.logger(
            "[#494b9b]Patched for tiling in the [#48a971]"
            + "X" * X
            + "[#494b9b] and [#48a971]" * (X and Y)
            + "Y" * Y
            + "[#494b9b] direction"
            + "s" * (X and Y)
        )

    return model, modelFS, modelPV
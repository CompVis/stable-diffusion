# Estimating and Exploiting the Aleatoric Uncertainty in Surface Normal Estimation
# https://github.com/baegwangbin/surface_normal_uncertainty

import types
import torch
import numpy as np

from einops import rearrange
from .models.NNET import NNET
from .utils import utils
import torchvision.transforms as transforms


class NormalBaeDetector:
    def __init__(self, modelpath):
        args = types.SimpleNamespace()
        args.mode = "client"
        args.architecture = "BN"
        args.pretrained = "scannet"
        args.sampling_ratio = 0.4
        args.importance_ratio = 0.7
        model = NNET(args)
        model = utils.load_checkpoint(modelpath, model)
        model = model.cuda()
        model.eval()
        self.model = model
        self.norm = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def __call__(self, input_image):
        assert input_image.ndim == 3
        image_normal = input_image
        with torch.no_grad():
            image_normal = torch.from_numpy(image_normal).float().cuda()
            image_normal = image_normal / 255.0
            image_normal = rearrange(image_normal, "h w c -> 1 c h w")
            image_normal = self.norm(image_normal)

            normal = self.model(image_normal)
            normal = normal[0][-1][:, :3]
            normal = ((normal + 1) * 0.5).clip(0, 1)

            normal = rearrange(normal[0], "c h w -> h w c").cpu().numpy()
            normal_image = (normal * 255.0).clip(0, 255).astype(np.uint8)

            return normal_image

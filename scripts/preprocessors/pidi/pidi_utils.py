import os
import torch
import numpy as np
from einops import rearrange
from preprocessors.preprocessors_util import safe_step
from .model import pidinet

class PidiNetDetector:
    def __init__(self, model_path):
        self.netNetwork = pidinet()
        self.netNetwork.load_state_dict(
            {
                k.replace("module.", ""): v
                for k, v in torch.load(model_path)["state_dict"].items()
            }
        )
        self.netNetwork = self.netNetwork.cuda()
        self.netNetwork.eval()

    def __call__(self, input_image, safe=False):
        assert input_image.ndim == 3
        input_image = input_image[:, :, ::-1].copy()
        with torch.no_grad():
            image_pidi = torch.from_numpy(input_image).float().cuda()
            image_pidi = image_pidi / 255.0
            image_pidi = rearrange(image_pidi, "h w c -> 1 c h w")
            edge = self.netNetwork(image_pidi)[-1]
            edge = edge.cpu().numpy()
            if safe:
                edge = safe_step(edge)
            edge = (edge * 255.0).clip(0, 255).astype(np.uint8)
            return edge[0][0]

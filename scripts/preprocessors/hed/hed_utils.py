import os
import cv2
import torch
import numpy as np

from einops import rearrange


class DoubleConvBlock(torch.nn.Module):
    def __init__(self, input_channel, output_channel, layer_number):
        super().__init__()
        self.convs = torch.nn.Sequential()
        self.convs.append(
            torch.nn.Conv2d(
                in_channels=input_channel,
                out_channels=output_channel,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=1,
            )
        )
        for i in range(1, layer_number):
            self.convs.append(
                torch.nn.Conv2d(
                    in_channels=output_channel,
                    out_channels=output_channel,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=1,
                )
            )
        self.projection = torch.nn.Conv2d(
            in_channels=output_channel,
            out_channels=1,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=0,
        )

    def __call__(self, x, down_sampling=False):
        h = x
        if down_sampling:
            h = torch.nn.functional.max_pool2d(h, kernel_size=(2, 2), stride=(2, 2))
        for conv in self.convs:
            h = conv(h)
            h = torch.nn.functional.relu(h)
        return h, self.projection(h)


class ControlNetHED_Apache2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = torch.nn.Parameter(torch.zeros(size=(1, 3, 1, 1)))
        self.block1 = DoubleConvBlock(
            input_channel=3, output_channel=64, layer_number=2
        )
        self.block2 = DoubleConvBlock(
            input_channel=64, output_channel=128, layer_number=2
        )
        self.block3 = DoubleConvBlock(
            input_channel=128, output_channel=256, layer_number=3
        )
        self.block4 = DoubleConvBlock(
            input_channel=256, output_channel=512, layer_number=3
        )
        self.block5 = DoubleConvBlock(
            input_channel=512, output_channel=512, layer_number=3
        )

    def __call__(self, x):
        h = x - self.norm
        h, projection1 = self.block1(h)
        h, projection2 = self.block2(h, down_sampling=True)
        h, projection3 = self.block3(h, down_sampling=True)
        h, projection4 = self.block4(h, down_sampling=True)
        h, projection5 = self.block5(h, down_sampling=True)
        return projection1, projection2, projection3, projection4, projection5


def safe_step(x, step=2):
    y = x.astype(np.float32) * float(step + 1)
    y = y.astype(np.int32).astype(np.float32) / float(step)
    return y


class HEDdetector:
    def __init__(self, model_path):
        self.netNetwork = ControlNetHED_Apache2().float().cuda().eval()
        self.netNetwork.load_state_dict(torch.load(model_path))

    def __call__(self, input_image, safe=False):
        assert input_image.ndim == 3
        H, W, C = input_image.shape
        with torch.no_grad():
            image_hed = torch.from_numpy(input_image.copy()).float().cuda()
            image_hed = rearrange(image_hed, "h w c -> 1 c h w")
            edges = self.netNetwork(image_hed)
            edges = [e.detach().cpu().numpy().astype(np.float32)[0, 0] for e in edges]
            edges = [
                cv2.resize(e, (W, H), interpolation=cv2.INTER_LINEAR) for e in edges
            ]
            edges = np.stack(edges, axis=2)
            edge = 1 / (1 + np.exp(-np.mean(edges, axis=2).astype(np.float64)))
            if safe:
                edge = safe_step(edge)
            edge = (edge * 255.0).clip(0, 255).astype(np.uint8)
            return edge

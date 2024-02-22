'''
 * Written by W.J. van der Laan for Astropulse LLC
 * Copyright (C) Astropulse LLC - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Unauthorized modification, redistribution, or reconstruction for any personal or commercial purpose is strictly prohibited
 * Proprietary and confidential
'''
import torch, io
from torch import nn
from cryptography.fernet import Fernet
import random
import numpy as np

def conv(n_in, n_out, **kwargs):
    return nn.Conv2d(n_in, n_out, 3, padding=1, **kwargs)

class Block(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.conv = nn.Sequential(conv(n_in, n_out), nn.ReLU(), conv(n_out, n_out), nn.ReLU(), conv(n_out, n_out))
        self.skip = nn.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        self.fuse = nn.ReLU()
    def forward(self, x):
        return self.fuse(self.conv(x) + self.skip(x))

def Decoder(bin_sizes):
    return nn.Sequential(
        conv(4, 64), nn.ReLU(),
        Block(64, 64), conv(64, sum(bin_sizes)),
    )

def rgb_to_hsv(rgb: torch.Tensor) -> torch.Tensor:
    cmax, cmax_idx = torch.max(rgb, dim=1, keepdim=True)
    cmin = torch.min(rgb, dim=1, keepdim=True)[0]
    delta = cmax - cmin
    hsv_h = torch.empty_like(rgb[:, 0:1, :, :])
    cmax_idx[delta == 0] = 3
    hsv_h[cmax_idx == 0] = (((rgb[:, 1:2] - rgb[:, 2:3]) / delta) % 6)[cmax_idx == 0]
    hsv_h[cmax_idx == 1] = (((rgb[:, 2:3] - rgb[:, 0:1]) / delta) + 2)[cmax_idx == 1]
    hsv_h[cmax_idx == 2] = (((rgb[:, 0:1] - rgb[:, 1:2]) / delta) + 4)[cmax_idx == 2]
    hsv_h[cmax_idx == 3] = 0.
    hsv_h /= 6.
    hsv_s = torch.where(cmax == 0, torch.tensor(0.).type_as(rgb), delta / cmax)
    hsv_v = cmax
    return torch.cat([hsv_h, hsv_s, hsv_v], dim=1)

def hsv_to_rgb(hsv: torch.Tensor) -> torch.Tensor:
    hsv_h, hsv_s, hsv_l = hsv[:, 0:1], hsv[:, 1:2], hsv[:, 2:3]
    _c = hsv_l * hsv_s
    _x = _c * (- torch.abs(hsv_h * 6. % 2. - 1) + 1.)
    _m = hsv_l - _c
    _o = torch.zeros_like(_c)
    idx = (hsv_h * 6.).type(torch.uint8)
    idx = (idx % 6).expand(-1, 3, -1, -1)
    rgb = torch.empty_like(hsv)
    rgb[idx == 0] = torch.cat([_c, _x, _o], dim=1)[idx == 0]
    rgb[idx == 1] = torch.cat([_x, _c, _o], dim=1)[idx == 1]
    rgb[idx == 2] = torch.cat([_o, _c, _x], dim=1)[idx == 2]
    rgb[idx == 3] = torch.cat([_o, _x, _c], dim=1)[idx == 3]
    rgb[idx == 4] = torch.cat([_x, _o, _c], dim=1)[idx == 4]
    rgb[idx == 5] = torch.cat([_c, _o, _x], dim=1)[idx == 5]
    rgb += _m
    return rgb

class HSVCube:
    def __init__(self, h, s, v):
        self.bins_per_channel = [h, s, v]

    def rgb8_to_cats(self, color, use_ignore_index):
        rgb = color / 255.0

        hsv = rgb_to_hsv(rgb)

        # quantize
        cat_h = torch.clamp(hsv[:, 0, :, :] * self.bins_per_channel[0], 0.0, self.bins_per_channel[0] - 1).to(dtype=torch.long)
        cat_s = torch.clamp(hsv[:, 1, :, :] * self.bins_per_channel[1], 0.0, self.bins_per_channel[1] - 1).to(dtype=torch.long)
        cat_v = torch.clamp(hsv[:, 2, :, :] * self.bins_per_channel[2], 0.0, self.bins_per_channel[2] - 1).to(dtype=torch.long)

        if use_ignore_index:
            # for CrossEntropyLoss's ignore_index=... 
            s_zero = cat_s == 0
            v_zero = cat_v == 0
            # if v=0, color is always black so s and h don't matter
            # if s=0,  color is always a grey so h doesn't matter (but v does)
            cat_h[v_zero | s_zero] = -1
            cat_s[v_zero] = -1

        return torch.stack([cat_h, cat_s, cat_v], dim=1)

    def cats_to_rgb8(self, cats):
        cat_h = cats[:, 0, :, :]
        cat_s = cats[:, 1, :, :]
        cat_v = cats[:, 2, :, :]

        # de-quantize
        # for h we use N instead of N-1, because 1.0 is the same as 0.0
        # there's no point in having two buckets encode for the same hue, so make return values in the middle of the bucket
        hsv = torch.stack([(cat_h + 0.5) / self.bins_per_channel[0], cat_s / (self.bins_per_channel[1] - 1.0), cat_v / (self.bins_per_channel[2] - 1.0)], dim=1)
        rgb = hsv_to_rgb(hsv)

        # clamp and convert to bytes
        rgb = torch.clamp(rgb * 255.0, 0.0, 255.0)
        return rgb.to(dtype=torch.uint8)

def net_output_to_color8(binning, predicts):
    # Net output to color, unrestricted palette.
    bin_idx = 0
    best = torch.zeros([predicts.size(0), len(binning.bins_per_channel), predicts.size(2), predicts.size(3)], dtype=torch.long)
    for channel_idx, chan_bins in enumerate(binning.bins_per_channel):
        best[:, channel_idx, :, :] = torch.argmax(predicts[:, bin_idx:bin_idx + chan_bins, :, :], 1)
        bin_idx += chan_bins

    image = binning.cats_to_rgb8(best)
    image = image.permute(0, 2, 3, 1)
    return image


def compute_softmax(binning, predicts):
    # compute softmax for each bin to get probability distribution in 0..1 for each channel
    bin_idx = 0
    for channel_idx, chan_bins in enumerate(binning.bins_per_channel):
        predicts[:, bin_idx:bin_idx + chan_bins, :, :] = torch.softmax(predicts[:, bin_idx:bin_idx + chan_bins, :, :], 1)
        bin_idx += chan_bins


def net_output_to_color8_pal(binning, predicts, pal):
    # Net output to color, correlating to limited palette.
    pal = torch.tensor(pal).reshape((-1, 3))

    # permute palette into wide image, then gets bins for every palette entry
    pal_img = pal.permute(1, 0)[None, :, None, :]
    pal_cats = binning.rgb8_to_cats(pal_img, False)
    pal_bins = pal_cats[0, :, 0, :].permute(1, 0)

    # convert palette itself to uint8[3] and transfer to device
    pal = pal.type(torch.uint8).to(predicts.device)

    # determine bins indices to sample
    bin_idx = 0
    for channel_idx, chan_bins in enumerate(binning.bins_per_channel):
        pal_bins[:, channel_idx] += bin_idx
        bin_idx += chan_bins

    compute_softmax(binning, predicts)

    # look up the three bins associated with the palette color for each image position
    extended = predicts[:, pal_bins, :, :] # [1, 4, 3, 64, 64]
    # multiply probabilities for channels
    extended = torch.prod(extended, 2)
    # get the entry with the highest product
    best = torch.argmax(extended, 1)

    image = pal[best]
    return image

NEIGHBOURHOOD8 = [(-1, -1), (-1, 0), (-1, 1), ( 0, -1), ( 0, 1), ( 1, -1), ( 1, 0), ( 1, 1)]
NEIGHBOURHOOD4 = [(-1, 0), ( 0, -1), ( 0, 1), ( 1, 0)]

class PixelVAE:
    def __init__(self, device, model, binning):
        self.device = device
        self.model = model
        self.binning = binning

    def to(self, device):
        self.device = device
        self.model.to(device)

    def run_plain(self, samples):
        predicts = self.model.forward(samples.to(self.device).to(torch.float32))
        result = net_output_to_color8(self.binning, predicts)
        return result

    def run_paletted(self, samples, palette):
        predicts = self.model.forward(samples.to(self.device).to(torch.float32))
        result = net_output_to_color8_pal(self.binning, predicts, palette)
        return result

    def run_cluster(self, samples, threshold=0.001, rand_seed=1, select='local8', wrap_x=False, wrap_y=False):
        predicts = self.model.forward(samples.to(self.device).to(torch.float32)).to(torch.float32)

        # compute softmax per channel, split channels
        compute_softmax(self.binning, predicts)
        bins = []
        bin_idx = 0
        for channel_idx, chan_bins in enumerate(self.binning.bins_per_channel):
            bins.append(predicts[0, bin_idx:bin_idx + chan_bins, :, :].cpu().numpy())
            bin_idx += chan_bins

        # do clustering
        height = bins[0].shape[1]
        width = bins[0].shape[2]

        cats = torch.zeros((3, height, width), dtype=torch.uint8)

        # predictable generator to shuffle visiting order
        visit_order = list((x, y) for y in range(height) for x in range(width))
        generator = random.Random(rand_seed)
        generator.shuffle(visit_order)

        # keep track of visited pixels
        visited = torch.zeros((height, width), dtype=torch.uint8)

        # precompute maximum product for each pixel
        maxprod = np.max(bins[0], axis=0) * np.max(bins[1], axis=0) * np.max(bins[2], axis=0)
        # multiply with the relative threshold to get per-pixel threshold
        pixel_threshold = maxprod * threshold

        def flood_fill(x, y):
            queue = [(x, y)]
            while queue:
                (x, y) = queue.pop()
                for (xi, yi) in neighbourhood:
                    xp, yp = x + xi, y + yi
                    if xp < 0 or yp < 0 or xp >= width or yp >= height:
                        # handle wrap-around boundary conditions, if requested
                        if wrap_x:
                            if xp < 0:
                                xp += width
                            elif xp >= width:
                                xp -= width
                        if wrap_y:
                            if yp < 0:
                                yp += height
                            elif yp >= height:
                                yp -= height
                        # if still out of bounds, skip pixel
                        if xp < 0 or yp < 0 or xp >= width or yp >= height:
                            continue
                    hsub, ssub, vsub = bins[0][hh,yp,xp], bins[1][ss,yp,xp], bins[2][vv,yp,xp]

                    if not visited[yp, xp] and hsub * ssub * vsub >= pixel_threshold[yp,xp]:
                        cats[:,yp,xp] = torch.Tensor([hh, ss, vv])
                        visited[yp, xp] = True
                        queue.append((xp, yp))

        def select_global(x, y):
            for yp in range(height):
                for xp in range(width):
                    hsub, ssub, vsub = bins[0][hh,yp,xp], bins[1][ss,yp,xp], bins[2][vv,yp,xp]

                    if not visited[yp, xp] and hsub * ssub * vsub >= pixel_threshold[yp,xp]:
                        cats[:,yp,xp] = torch.Tensor([hh, ss, vv])
                        visited[yp, xp] = True

        if select == 'local8':
            visit = flood_fill
            neighbourhood = NEIGHBOURHOOD8
        elif select == 'local4':
            visit = flood_fill
            neighbourhood = NEIGHBOURHOOD4
        elif select == 'global':
            visit = select_global
        else:
            raise NotImplemented

        # random sampling
        for (x, y) in visit_order:
            if visited[y, x]:
                continue

            hh = np.argmax(bins[0][:,y,x])
            ss = np.argmax(bins[1][:,y,x])
            vv = np.argmax(bins[2][:,y,x])

            cats[:,y,x] = torch.Tensor([hh, ss, vv])
            visited[y, x] = True

            visit(x, y)

        result = self.binning.cats_to_rgb8(cats[None])
        result = result.permute(0, 2, 3, 1)
        return result

def load_pixelvae_model(weights_path, device, key):
    fernet = Fernet(key)
    with open(weights_path, 'rb') as enc_file:
        encrypted = enc_file.read()

    decryptedStream = io.BytesIO(fernet.decrypt(encrypted))

    binning = HSVCube(32, 8, 16)

    model = Decoder(binning.bins_per_channel)
    state_dict = torch.load(decryptedStream, map_location=device)
    model.load_state_dict(state_dict['state_dict'])
    model.eval()
    model = model.to(device)

    return PixelVAE(device, model, binning)
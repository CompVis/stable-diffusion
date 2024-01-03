import random

import cv2
import numpy as np
from preprocessors.preprocessors_util import img2mask, make_noise_disk

class ContentShuffleDetector:
    def __call__(self, img, h=None, w=None, f=None):
        H, W, C = img.shape
        if h is None:
            h = H
        if w is None:
            w = W
        if f is None:
            f = 256
        x = make_noise_disk(h, w, 1, f) * float(W - 1)
        y = make_noise_disk(h, w, 1, f) * float(H - 1)
        flow = np.concatenate([x, y], axis=2).astype(np.float32)
        return cv2.remap(img, flow, None, cv2.INTER_LINEAR)


class ColorShuffleDetector:
    def __call__(self, img):
        H, W, C = img.shape
        F = random.randint(64, 384)
        A = make_noise_disk(H, W, 3, F)
        B = make_noise_disk(H, W, 3, F)
        C = (A + B) / 2.0
        A = (C + (A - C) * 3.0).clip(0, 1)
        B = (C + (B - C) * 3.0).clip(0, 1)
        L = img.astype(np.float32) / 255.0
        Y = A * L + B * (1 - L)
        Y -= np.min(Y, axis=(0, 1), keepdims=True)
        Y /= np.maximum(np.max(Y, axis=(0, 1), keepdims=True), 1e-5)
        Y *= 255.0
        return Y.clip(0, 255).astype(np.uint8)


class GrayDetector:
    def __call__(self, img):
        eps = 1e-5
        X = img.astype(np.float32)
        r, g, b = X[:, :, 0], X[:, :, 1], X[:, :, 2]
        kr, kg, kb = [random.random() + eps for _ in range(3)]
        ks = kr + kg + kb
        kr /= ks
        kg /= ks
        kb /= ks
        Y = r * kr + g * kg + b * kb
        Y = np.stack([Y] * 3, axis=2)
        return Y.clip(0, 255).astype(np.uint8)


class DownSampleDetector:
    def __call__(self, img, level=3, k=16.0):
        h = img.astype(np.float32)
        for _ in range(level):
            h += np.random.normal(loc=0.0, scale=k, size=h.shape)
            h = cv2.pyrDown(h)
        for _ in range(level):
            h = cv2.pyrUp(h)
            h += np.random.normal(loc=0.0, scale=k, size=h.shape)
        return h.clip(0, 255).astype(np.uint8)


class Image2MaskShuffleDetector:
    def __init__(self, resolution=(640, 512)):
        self.H, self.W = resolution

    def __call__(self, img):
        m = img2mask(img, self.H, self.W)
        m *= 255.0
        return m.clip(0, 255).astype(np.uint8)

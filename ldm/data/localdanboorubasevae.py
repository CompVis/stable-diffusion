import os
import numpy as np
import PIL
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF

from functools import partial
import copy

import glob

import random

PIL.Image.MAX_IMAGE_PIXELS = 933120000
import torchvision

import pytorch_lightning as pl

import torch

import re
import json
import io

def resize_image(image: Image, max_size=(768,768)):
    image = ImageOps.contain(image, max_size, Image.LANCZOS)
    # resize to integer multiple of 64
    w, h = image.size
    w, h = map(lambda x: x - x % 64, (w, h))

    ratio = w / h
    src_ratio = image.width / image.height

    src_w = w if ratio > src_ratio else image.width * h // image.height
    src_h = h if ratio <= src_ratio else image.height * w // image.width

    resized = image.resize((src_w, src_h), resample=Image.LANCZOS)
    res = Image.new("RGB", (w, h))
    res.paste(resized, box=(w // 2 - src_w // 2, h // 2 - src_h // 2))

    return res

class CaptionProcessor(object):
    def __init__(self, transforms, max_size, resize, random_order, LR_size):
        self.transforms = transforms
        self.max_size = max_size
        self.resize = resize
        self.random_order = random_order
        self.degradation_process = partial(TF.resize, size=LR_size, interpolation=TF.InterpolationMode.NEAREST)
    
    def __call__(self, sample):
        # preprocess caption
        pass

        # preprocess image
        image = sample['image']
        image = Image.open(io.BytesIO(image))
        if self.resize:
            image = resize_image(image, max_size=(self.max_size, self.max_size))
        image = self.transforms(image)
        lr_image = copy.deepcopy(image)
        image = np.array(image).astype(np.uint8)
        sample['image'] = (image / 127.5 - 1.0).astype(np.float32)

        # preprocess LR image
        lr_image = self.degradation_process(lr_image)
        lr_image = np.array(lr_image).astype(np.uint8)
        sample['LR_image'] = (lr_image/127.5 - 1.0).astype(np.float32)

        return sample

class LocalDanbooruBaseVAE(Dataset):
    def __init__(self,
                 data_root='./danbooru-aesthetic',
                 size=256,
                 interpolation="bicubic",
                 flip_p=0.5,
                 crop=True,
                 shuffle=False,
                 mode='train',
                 val_split=64,
                 downscale_f=8
                 ):
        super().__init__()

        self.shuffle=shuffle
        self.crop = crop

        print('Fetching data.')

        ext = ['image']
        self.image_files = []
        [self.image_files.extend(glob.glob(f'{data_root}' + '/*.' + e)) for e in ext]
        if mode == 'val':
            self.image_files = self.image_files[:len(self.image_files)//val_split]

        print(f'Constructing image map. Found {len(self.image_files)} images')

        self.examples = {}
        self.hashes = []
        for i in self.image_files:
            hash = i[len(f'{data_root}/'):].split('.')[0]
            self.examples[hash] = {
                'image': i
            }
            self.hashes.append(hash)

        print(f'image map has {len(self.examples.keys())} examples')

        self.size = size
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)

        image_transforms = []
        image_transforms.extend([torchvision.transforms.RandomHorizontalFlip(flip_p)],)
        image_transforms = torchvision.transforms.Compose(image_transforms)

        self.captionprocessor = CaptionProcessor(image_transforms, self.size, True, True, int(size / downscale_f))

    def random_sample(self):
        return self.__getitem__(random.randint(0, self.__len__() - 1))
    
    def sequential_sample(self, i):
        if i >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(i + 1)

    def skip_sample(self, i):
        return None

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, i):
        return self.get_image(i)
    
    def get_image(self, i):
        image = {}
        try:
            image_file = self.examples[self.hashes[i]]['image']
            with open(image_file, 'rb') as f:
                image['image'] = f.read()
            image = self.captionprocessor(image)
        except Exception as e:
            print(f'Error with {self.examples[self.hashes[i]]["image"]} -- {e} -- skipping {i}')
            return self.skip_sample(i)
        
        return image

"""
if __name__ == "__main__":
    dataset = LocalBase('./danbooru-aesthetic', size=512, crop=False, mode='val')
    print(dataset.__len__())
    example = dataset.__getitem__(0)
    print(dataset.hashes[0])
    print(example['caption'])
    image = example['image']
    image = ((image + 1) * 127.5).astype(np.uint8)
    image = Image.fromarray(image)
    image.save('example.png')
"""
"""
from tqdm import tqdm
if __name__ == "__main__":
    dataset = LocalDanbooruBase('./links', size=768)
    import time
    a = time.process_time()
    for i in range(8):
        example = dataset.get_image(i)
        image = example['image']
        image = ((image + 1) * 127.5).astype(np.uint8)
        image = Image.fromarray(image)
        image.save(f'example-{i}.png')
        print(example['caption'])
    print('time:', time.process_time()-a)
"""

import os
import numpy as np
import PIL
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms

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
    def __init__(self, copyright_rate, character_rate, general_rate, artist_rate, normalize, caption_shuffle, transforms, max_size, resize, random_order):
        self.copyright_rate = copyright_rate
        self.character_rate = character_rate
        self.general_rate = general_rate
        self.artist_rate = artist_rate
        self.normalize = normalize
        self.caption_shuffle = caption_shuffle
        self.transforms = transforms
        self.max_size = max_size
        self.resize = resize
        self.random_order = random_order
    
    def clean(self, text: str):
        text = ' '.join(set([i.lstrip('_').rstrip('_') for i in re.sub(r'\([^)]*\)', '', text).split(' ')])).lstrip().rstrip()
        if self.caption_shuffle:
            text = text.split(' ')
            random.shuffle(text)
            text = ' '.join(text)
        if self.normalize:
            text = ', '.join([i.replace('_', ' ') for i in text.split(' ')]).lstrip(', ').rstrip(', ')
        return text

    def get_key(self, val_dict, key, clean_val = True, cond_drop = 0.0, prepend_space = False, append_comma = False):
        space = ' ' if prepend_space else ''
        comma = ',' if append_comma else ''
        if random.random() < cond_drop:
            if (key in val_dict) and val_dict[key]:
                if clean_val:
                    return space + self.clean(val_dict[key]) + comma
                else:
                    return space + val_dict[key] + comma
        return ''

    def __call__(self, sample):
        # preprocess caption
        caption_data = json.loads(sample['caption'])
        if not self.random_order:
            character = self.get_key(caption_data, 'tag_string_character', True, self.character_rate, False, True)
            copyright = self.get_key(caption_data, 'tag_string_copyright', True, self.copyright_rate, True, True)
            artist = self.get_key(caption_data, 'tag_string_artist', True, self.artist_rate, True, True)
            general = self.get_key(caption_data, 'tag_string_general', True, self.general_rate, True, False)
            tag_str = f'{character}{copyright}{artist}{general}'.lstrip().rstrip(',')
        else:
            character = self.get_key(caption_data, 'tag_string_character', False, self.character_rate, False)
            copyright = self.get_key(caption_data, 'tag_string_copyright', False, self.copyright_rate, True, False)
            artist = self.get_key(caption_data, 'tag_string_artist', False, self.artist_rate, True, False)
            general = self.get_key(caption_data, 'tag_string_general', False, self.general_rate, True, False)
            tag_str = self.clean(f'{character}{copyright}{artist}{general}').lstrip().rstrip(' ')
        sample['caption'] = tag_str

        # preprocess image
        image = sample['image']
        image = Image.open(io.BytesIO(image))
        if self.resize:
            image = resize_image(image, max_size=(self.max_size, self.max_size))
        image = self.transforms(image)
        image = np.array(image).astype(np.uint8)
        sample['image'] = (image / 127.5 - 1.0).astype(np.float32)
        return sample

class LocalDanbooruBase(Dataset):
    def __init__(self,
                 data_root='./danbooru-aesthetic',
                 size=768,
                 interpolation="bicubic",
                 flip_p=0.5,
                 crop=True,
                 shuffle=False,
                 mode='train',
                 val_split=64,
                 ucg=0.1,
                 ):
        super().__init__()

        self.shuffle=shuffle
        self.crop = crop
        self.ucg = ucg

        print('Fetching data.')

        ext = ['image']
        self.image_files = []
        [self.image_files.extend(glob.glob(f'{data_root}' + '/*.' + e)) for e in ext]
        if mode == 'val':
            self.image_files = self.image_files[:len(self.image_files)//val_split]

        print(f'Constructing image-caption map. Found {len(self.image_files)} images')

        self.examples = {}
        self.hashes = []
        for i in self.image_files:
            hash = i[len(f'{data_root}/'):].split('.')[0]
            self.examples[hash] = {
                'image': i,
                'text': f'{data_root}/{hash}.caption'
            }
            self.hashes.append(hash)

        print(f'image-caption map has {len(self.examples.keys())} examples')

        self.size = size
        self.interpolation = {"linear": PIL.Image.Resampling.BILINEAR,
                              "bilinear": PIL.Image.Resampling.BILINEAR,
                              "bicubic": PIL.Image.Resampling.BICUBIC,
                              "lanczos": PIL.Image.Resampling.LANCZOS,
                              }[interpolation]
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)

        image_transforms = []
        image_transforms.extend([torchvision.transforms.RandomHorizontalFlip(flip_p)],)
        image_transforms = torchvision.transforms.Compose(image_transforms)

        self.captionprocessor = CaptionProcessor(1.0, 1.0, 1.0, 1.0, True, True, image_transforms, 768, False, True)

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
            text_file = self.examples[self.hashes[i]]['text']
            with open(text_file, 'rb') as f:
                image['caption'] = f.read()
            image = self.captionprocessor(image)
            if random.random() < self.ucg:
                image['caption'] = ''
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
import os
import numpy as np
import PIL
from PIL import Image, ImageOps
import random

PIL.Image.MAX_IMAGE_PIXELS = 933120000

import webdataset as wds
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

def dict_collation_fn(samples, combine_tensors=True, combine_scalars=True):
    """Take a list  of samples (as dictionary) and create a batch, preserving the keys.
    If `tensors` is True, `ndarray` objects are combined into
    tensor batches.
    :param dict samples: list of samples
    :param bool tensors: whether to turn lists of ndarrays into a single ndarray
    :returns: single sample consisting of a batch
    :rtype: dict
    """
    keys = set.intersection(*[set(sample.keys()) for sample in samples])
    batched = {key: [] for key in keys}

    for s in samples:
        [batched[key].append(s[key]) for key in batched]

    result = {}
    for key in batched:
        if isinstance(batched[key][0], (int, float)):
            if combine_scalars:
                result[key] = np.array(list(batched[key]))
        elif isinstance(batched[key][0], torch.Tensor):
            if combine_tensors:
                result[key] = torch.stack(list(batched[key]))
        elif isinstance(batched[key][0], np.ndarray):
            if combine_tensors:
                result[key] = np.array(list(batched[key]))
        else:
            result[key] = list(batched[key])
    return result


class DanbooruWebDataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, tar_base, batch_size, train=None, validation=None,
                 test=None, num_workers=4, max_size=768, resize=False, flip_p=0.5, image_key='image', copyright_rate=0.9, character_rate=0.9, general_rate=0.9, artist_rate=0.9, normalize=True, caption_shuffle=True, random_order=True,
                 **kwargs):
        super().__init__(self)
        print(f'Setting tar base to {tar_base}')
        self.tar_base = tar_base
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train = train
        self.validation = validation
        self.test = test
        self.max_size = max_size
        self.resize = resize
        self.flip_p = flip_p
        self.image_key = image_key
        self.copyright_rate = copyright_rate
        self.character_rate = character_rate
        self.general_rate = general_rate
        self.artist_rate = artist_rate
        self.normalize = normalize
        self.caption_shuffle = caption_shuffle
        self.random_order = random_order

    def make_loader(self, dataset_config, train=True):
        image_transforms = []
        image_transforms.extend([torchvision.transforms.RandomHorizontalFlip(self.flip_p)],)
        image_transforms = torchvision.transforms.Compose(image_transforms)

        transform_dict = {}
        transform_dict.update({self.image_key: image_transforms})

        postprocess = CaptionProcessor(copyright_rate=self.copyright_rate, character_rate=self.character_rate, general_rate=self.general_rate, artist_rate=self.artist_rate, normalize=self.normalize, caption_shuffle=self.caption_shuffle, transforms=image_transforms, max_size=self.max_size, resize=self.resize, random_order=self.random_order)
        

        tars = os.path.join(self.tar_base)

        dset = wds.WebDataset(
                tars,
                handler=wds.warn_and_continue).repeat().shuffle(1.0)
        print(f'Loading webdataset with {len(dset.pipeline[0].urls)} shards.')
        dset = (dset
                .select(self.filter_keys)
                )
        if postprocess is not None:
            dset = dset.map(postprocess)
        dset = (dset
                .batched(self.batch_size, partial=False,
                    collation_fn=dict_collation_fn)
                )

        loader = wds.WebLoader(dset, batch_size=None, shuffle=False,
                               num_workers=self.num_workers)

        return loader

    def filter_keys(self, x):
        return True

    def train_dataloader(self):
        return self.make_loader(self.train)

    def val_dataloader(self):
        return self.make_loader(self.validation, train=False)

    def test_dataloader(self):
        return self.make_loader(self.test, train=False)

def example():
    from omegaconf import OmegaConf
    from torch.utils.data.distributed import DistributedSampler
    from torch.utils.data import IterableDataset
    from torch.utils.data import DataLoader, RandomSampler, Sampler, SequentialSampler
    from pytorch_lightning.trainer.supporters import CombinedLoader, CycleIterator

    config = OmegaConf.load("configs/stable-diffusion/v1-finetune-danbooru-8gpu.yaml")
    datamod = DanbooruWebDataModuleFromConfig(**config["data"]["params"])
    dataloader = datamod.train_dataloader()

    for batch in dataloader:
        print(batch["image"].shape)
        print(batch['caption'])
        image = ((batch["image"][0] + 1) * 127.5).numpy().astype(np.uint8)
        image = Image.fromarray(image)
        image.save('example.png')
        break

if __name__ == '__main__':
    #example()
    pass
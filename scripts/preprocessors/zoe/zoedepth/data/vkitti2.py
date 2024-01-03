# MIT License

# Copyright (c) 2022 Intelligent Systems Lab Org

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# File author: Shariq Farooq Bhat

import os

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class ToTensor(object):
    def __init__(self):
        # self.normalize = transforms.Normalize(
        #     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.normalize = lambda x: x
        # self.resize = transforms.Resize((375, 1242))

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']

        image = self.to_tensor(image)
        image = self.normalize(image)
        depth = self.to_tensor(depth)

        # image = self.resize(image)

        return {'image': image, 'depth': depth, 'dataset': "vkitti"}

    def to_tensor(self, pic):

        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img

        #         # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(
                torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)

        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float()
        else:
            return img


class VKITTI2(Dataset):
    def __init__(self, data_dir_root, do_kb_crop=True, split="test"):
        import glob

        # image paths are of the form <data_dir_root>/rgb/<scene>/<variant>/frames/<rgb,depth>/Camera<0,1>/rgb_{}.jpg
        self.image_files = glob.glob(os.path.join(
            data_dir_root, "rgb", "**", "frames", "rgb", "Camera_0", '*.jpg'), recursive=True)
        self.depth_files = [r.replace("/rgb/", "/depth/").replace(
            "rgb_", "depth_").replace(".jpg", ".png") for r in self.image_files]
        self.do_kb_crop = True
        self.transform = ToTensor()

        # If train test split is not created, then create one.
        # Split is such that 8% of the frames from each scene are used for testing.
        if not os.path.exists(os.path.join(data_dir_root, "train.txt")):
            import random
            scenes = set([os.path.basename(os.path.dirname(
                os.path.dirname(os.path.dirname(f)))) for f in self.image_files])
            train_files = []
            test_files = []
            for scene in scenes:
                scene_files = [f for f in self.image_files if os.path.basename(
                    os.path.dirname(os.path.dirname(os.path.dirname(f)))) == scene]
                random.shuffle(scene_files)
                train_files.extend(scene_files[:int(len(scene_files) * 0.92)])
                test_files.extend(scene_files[int(len(scene_files) * 0.92):])
            with open(os.path.join(data_dir_root, "train.txt"), "w") as f:
                f.write("\n".join(train_files))
            with open(os.path.join(data_dir_root, "test.txt"), "w") as f:
                f.write("\n".join(test_files))

        if split == "train":
            with open(os.path.join(data_dir_root, "train.txt"), "r") as f:
                self.image_files = f.read().splitlines()
            self.depth_files = [r.replace("/rgb/", "/depth/").replace(
                "rgb_", "depth_").replace(".jpg", ".png") for r in self.image_files]
        elif split == "test":
            with open(os.path.join(data_dir_root, "test.txt"), "r") as f:
                self.image_files = f.read().splitlines()
            self.depth_files = [r.replace("/rgb/", "/depth/").replace(
                "rgb_", "depth_").replace(".jpg", ".png") for r in self.image_files]

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        depth_path = self.depth_files[idx]

        image = Image.open(image_path)
        # depth = Image.open(depth_path)
        depth = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR |
                           cv2.IMREAD_ANYDEPTH) / 100.0  # cm to m
        depth = Image.fromarray(depth)
        # print("dpeth min max", depth.min(), depth.max())

        # print(np.shape(image))
        # print(np.shape(depth))

        if self.do_kb_crop:
            if idx == 0:
                print("Using KB input crop")
            height = image.height
            width = image.width
            top_margin = int(height - 352)
            left_margin = int((width - 1216) / 2)
            depth = depth.crop(
                (left_margin, top_margin, left_margin + 1216, top_margin + 352))
            image = image.crop(
                (left_margin, top_margin, left_margin + 1216, top_margin + 352))
            # uv = uv[:, top_margin:top_margin + 352, left_margin:left_margin + 1216]

        image = np.asarray(image, dtype=np.float32) / 255.0
        # depth = np.asarray(depth, dtype=np.uint16) /1.
        depth = np.asarray(depth, dtype=np.float32) / 1.
        depth[depth > 80] = -1

        depth = depth[..., None]
        sample = dict(image=image, depth=depth)

        # return sample
        sample = self.transform(sample)

        if idx == 0:
            print(sample["image"].shape)

        return sample

    def __len__(self):
        return len(self.image_files)


def get_vkitti2_loader(data_dir_root, batch_size=1, **kwargs):
    dataset = VKITTI2(data_dir_root)
    return DataLoader(dataset, batch_size, **kwargs)


if __name__ == "__main__":
    loader = get_vkitti2_loader(
        data_dir_root="/home/bhatsf/shortcuts/datasets/vkitti2")
    print("Total files", len(loader.dataset))
    for i, sample in enumerate(loader):
        print(sample["image"].shape)
        print(sample["depth"].shape)
        print(sample["dataset"])
        print(sample['depth'].min(), sample['depth'].max())
        if i > 5:
            break

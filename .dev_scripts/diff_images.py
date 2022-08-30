import argparse

import numpy as np
from PIL import Image


def read_image_int16(image_path):
    image = Image.open(image_path)
    return np.array(image).astype(np.int16)


def calc_images_mean_L1(image1_path, image2_path):
    image1 = read_image_int16(image1_path)
    image2 = read_image_int16(image2_path)
    assert image1.shape == image2.shape

    mean_L1 = np.abs(image1 - image2).mean()
    return mean_L1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('image1_path')
    parser.add_argument('image2_path')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    mean_L1 = calc_images_mean_L1(args.image1_path, args.image2_path)
    print(mean_L1)

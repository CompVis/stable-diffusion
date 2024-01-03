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

import numpy as np
from dataclasses import dataclass
from typing import Tuple, List

# dataclass to store the crop parameters
@dataclass
class CropParams:
    top: int
    bottom: int
    left: int
    right: int



def get_border_params(rgb_image, tolerance=0.1, cut_off=20, value=0, level_diff_threshold=5, channel_axis=-1, min_border=5) -> CropParams:
    gray_image = np.mean(rgb_image, axis=channel_axis)
    h, w = gray_image.shape


    def num_value_pixels(arr):
        return np.sum(np.abs(arr - value) < level_diff_threshold)

    def is_above_tolerance(arr, total_pixels):
        return (num_value_pixels(arr) / total_pixels) > tolerance

    # Crop top border until number of value pixels become below tolerance
    top = min_border
    while is_above_tolerance(gray_image[top, :], w) and top < h-1:
        top += 1
        if top > cut_off:
            break

    # Crop bottom border until number of value pixels become below tolerance
    bottom = h - min_border
    while is_above_tolerance(gray_image[bottom, :], w) and bottom > 0:
        bottom -= 1
        if h - bottom > cut_off:
            break

    # Crop left border until number of value pixels become below tolerance
    left = min_border
    while is_above_tolerance(gray_image[:, left], h) and left < w-1:
        left += 1
        if left > cut_off:
            break

    # Crop right border until number of value pixels become below tolerance
    right = w - min_border
    while is_above_tolerance(gray_image[:, right], h) and right > 0:
        right -= 1
        if w - right > cut_off:
            break
        

    return CropParams(top, bottom, left, right)


def get_white_border(rgb_image, value=255, **kwargs) -> CropParams:
    """Crops the white border of the RGB.

    Args:
        rgb: RGB image, shape (H, W, 3).
    Returns:
        Crop parameters.
    """
    if value == 255:
        # assert range of values in rgb image is [0, 255]
        assert np.max(rgb_image) <= 255 and np.min(rgb_image) >= 0, "RGB image values are not in range [0, 255]."
        assert rgb_image.max() > 1, "RGB image values are not in range [0, 255]."
    elif value == 1:
        # assert range of values in rgb image is [0, 1]
        assert np.max(rgb_image) <= 1 and np.min(rgb_image) >= 0, "RGB image values are not in range [0, 1]."

    return get_border_params(rgb_image, value=value, **kwargs)

def get_black_border(rgb_image, **kwargs) -> CropParams:
    """Crops the black border of the RGB.

    Args:
        rgb: RGB image, shape (H, W, 3).

    Returns:
        Crop parameters.
    """

    return get_border_params(rgb_image, value=0, **kwargs)

def crop_image(image: np.ndarray, crop_params: CropParams) -> np.ndarray:
    """Crops the image according to the crop parameters.

    Args:
        image: RGB or depth image, shape (H, W, 3) or (H, W).
        crop_params: Crop parameters.

    Returns:
        Cropped image.
    """
    return image[crop_params.top:crop_params.bottom, crop_params.left:crop_params.right]

def crop_images(*images: np.ndarray, crop_params: CropParams) -> Tuple[np.ndarray]:
    """Crops the images according to the crop parameters.

    Args:
        images: RGB or depth images, shape (H, W, 3) or (H, W).
        crop_params: Crop parameters.

    Returns:
        Cropped images.
    """
    return tuple(crop_image(image, crop_params) for image in images)

def crop_black_or_white_border(rgb_image, *other_images: np.ndarray, tolerance=0.1, cut_off=20, level_diff_threshold=5) -> Tuple[np.ndarray]:
    """Crops the white and black border of the RGB and depth images.

    Args:
        rgb: RGB image, shape (H, W, 3). This image is used to determine the border.
        other_images: The other images to crop according to the border of the RGB image.
    Returns:
        Cropped RGB and other images.
    """
    # crop black border
    crop_params = get_black_border(rgb_image, tolerance=tolerance, cut_off=cut_off, level_diff_threshold=level_diff_threshold)
    cropped_images = crop_images(rgb_image, *other_images, crop_params=crop_params)

    # crop white border
    crop_params = get_white_border(cropped_images[0], tolerance=tolerance, cut_off=cut_off, level_diff_threshold=level_diff_threshold)
    cropped_images = crop_images(*cropped_images, crop_params=crop_params)

    return cropped_images
    
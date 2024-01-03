import cv2
from PIL import Image
from .hed.hed_utils import HEDdetector
from .preprocessors_util import HWC3

def canny(image: Image, low_threshold: float, high_threshold: float) -> Image:
    return cv2.Canny(image, low_threshold, high_threshold)

def hed(image: Image, model_path: str) -> Image:
    model = HEDdetector(model_path)
    return model(HWC3(image))
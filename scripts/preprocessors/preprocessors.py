import cv2
from PIL import Image

from .hed.hed_utils import HEDdetector
from .pidi.pidi_utils import PidiNetDetector
from .mlsd.mlsd_utils import MLSDdetector
from .zoe.zoe_utils import ZoeDetector
from .normalbae.normalbae_utils import NormalBaeDetector
from .openpose.openpose_utils import OpenposeDetector
from .shuffle.shuffle_utils import ContentShuffleDetector, ColorShuffleDetector

from .preprocessors_util import HWC3


def canny(image: Image, low_threshold: float, high_threshold: float) -> Image:
    return cv2.Canny(image, low_threshold, high_threshold)


def hed(image: Image, model_path: str) -> Image:
    model = HEDdetector(model_path)
    return model(HWC3(image))


def pidi(image: Image, model_path: str) -> Image:
    model = PidiNetDetector(model_path)
    return model(HWC3(image))


def mlsd(image: Image, thr_v: float, thr_d: float, model_path: str) -> Image:
    model = MLSDdetector(model_path)
    return model(HWC3(image), thr_v, thr_d)


def zoe(image: Image, model_path: str) -> Image:
    model = ZoeDetector(model_path)
    return model(HWC3(image))


def nornmalbae(image: Image, model_path: str) -> Image:
    model = NormalBaeDetector(model_path)
    return model(HWC3(image))


def openpose(
    image: Image, hand_and_face: bool, body_path: str, hand_path: str, face_path: str
) -> Image:
    model = OpenposeDetector(body_path, hand_path, face_path)
    return model(HWC3(image), hand_and_face)

def content_shuffle(image: Image) -> Image:
    model = ContentShuffleDetector()
    return model(HWC3(image))

def color_shuffle(image: Image) -> Image:
    model = ColorShuffleDetector()
    return model(HWC3(image))
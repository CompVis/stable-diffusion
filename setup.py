import os

import pkg_resources
from setuptools import setup, find_packages

setup(
    name="stablediffusion",
    py_modules=["stablediffusion"],
    version="0.3.0",
    description="",
    author="w4ffl35 (Joe Curlee)",
    packages=find_packages(exclude=["tests*"]),
    install_requires=[
        "torch",
        "numpy",
        "tqdm",
        "psutil",
        "opencv-python",
        "pudb",
        "invisible-watermark",
        "imageio",
        "imageio-ffmpeg",
        "pytorch-lightning",
        "omegaconf",
        "test-tube",
        "streamlit",
        "einops",
        "torch-fidelity",
        "transformers",
        "torchmetrics",
        "kornia",
        "natsort",
        "taming-transformers @ git+https://github.com/w4ffl35/taming-transformers.git#egg=taming",
        "clip @ git+https://github.com/w4ffl35/CLIP.git#egg=clip",
        "albumentations @ git+https://github.com/albumentations-team/albumentations#egg=albumentations",
    ],
    include_package_data=True,
    extras_require={'dev': ['pytest']},
)

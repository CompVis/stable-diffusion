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
        "natsort"
    ],
    include_package_data=True,
    extras_require={'dev': ['pytest']},
)

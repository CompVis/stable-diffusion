from setuptools import setup, find_packages

setup(
    name='k-diffusion',
    version='0.0.1',
    description='Karras et al. (2022) diffusion models for PyTorch',
    packages=find_packages(),
    install_requires=[
        'accelerate',
        'clean-fid',
        'einops',
        'jsonmerge',
        'kornia',
        'Pillow',
        'resize-right',
        'scikit-image',
        'scipy',
        'torch',
        'torchdiffeq',
        'torchvision',
        'tqdm',
        'wandb',
    ],
)

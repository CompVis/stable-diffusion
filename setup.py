from setuptools import setup, find_packages

setup(
    name='stable-diffusion',
    version='1.15.0-dev',
    description='',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'tqdm',
    ],
)
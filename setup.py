from setuptools import setup, find_packages

setup(
    name='invoke-ai',
    version='2.0.2',
    description='',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'tqdm',
    ],
)

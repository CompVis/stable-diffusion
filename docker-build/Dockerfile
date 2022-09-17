FROM        debian 

ARG         gsd
ENV         GITHUB_STABLE_DIFFUSION $gsd

ARG         rsd
ENV         REQS $rsd

ARG         cs
ENV         CONDA_SUBDIR $cs

ENV         PIP_EXISTS_ACTION="w"

# TODO: Optimize image size

SHELL       ["/bin/bash", "-c"]

WORKDIR     /
RUN         apt update && apt upgrade -y \
            && apt install -y \
            git \
            libgl1-mesa-glx \
            libglib2.0-0 \
            pip \
            python3 \
            && git clone $GITHUB_STABLE_DIFFUSION

# Install Anaconda or Miniconda
COPY        anaconda.sh .
RUN         bash anaconda.sh -b -u -p /anaconda && /anaconda/bin/conda init bash

# SD 
WORKDIR     /stable-diffusion 
RUN         source ~/.bashrc \
            && conda create -y --name ldm && conda activate ldm \
            && conda config --env --set subdir $CONDA_SUBDIR \
            && pip3 install -r $REQS \   
            && pip3 install basicsr facexlib realesrgan \          
            && mkdir models/ldm/stable-diffusion-v1 \
            && ln -s "/data/sd-v1-4.ckpt" models/ldm/stable-diffusion-v1/model.ckpt 

# Face restoreation
# by default expected in a sibling directory to stable-diffusion
WORKDIR     /       
RUN         git clone https://github.com/TencentARC/GFPGAN.git 

WORKDIR     /GFPGAN 
RUN         pip3 install -r requirements.txt \
            && python3 setup.py develop \
            && ln -s "/data/GFPGANv1.3.pth" experiments/pretrained_models/GFPGANv1.3.pth

WORKDIR     /stable-diffusion 
RUN         python3 scripts/preload_models.py         

WORKDIR     /
COPY        entrypoint.sh .
ENTRYPOINT ["/entrypoint.sh"]
FROM        debian 

ARG         gsd
ENV         GITHUB_STABLE_DIFFUSION $gsd

ARG         sdreq="requirements-linux-arm64.txt"
ENV         SD_REQ $sdreq

ARG         condaarch
ENV         ARCH $condaarch

WORKDIR     /

# TODO: Optimize image size

COPY        entrypoint.sh anaconda.sh .
SHELL       ["/bin/bash", "-c"]

# Update and apt 446 MB
RUN         apt update && apt upgrade -y \
            && apt install -y \
            git \
            pip \
            python3 \
            wget 

# install Anaconda or Miniconda 610 MB
RUN         bash anaconda.sh -b -u -p /anaconda && /anaconda/bin/conda init bash

# SD repo 105 MB
RUN         git clone $GITHUB_STABLE_DIFFUSION 

WORKDIR     /stable-diffusion 

# SD env 2.3 GB !!!
RUN         PIP_EXISTS_ACTION="w" \
            && CONDA_SUBDIR=$ARCH \
            && source ~/.bashrc && conda create -y --name ldm && conda activate ldm \
            && conda config --env --set subdir $ARCH \
            && pip3 install -r $SD_REQ \                     
            && mkdir models/ldm/stable-diffusion-v1 

# Face restoration prerequisites 200 MB
RUN         apt install -y libgl1-mesa-glx libglib2.0-0 

WORKDIR     /       

# Face restoreation repo 12 MB
# by default expected in a sibling directory to stable-diffusion
RUN         git clone https://github.com/TencentARC/GFPGAN.git 

WORKDIR     /GFPGAN 

# Face restoration env 608 MB
RUN         pip3 install basicsr facexlib \
            && pip3 install -r requirements.txt \
            && python3 setup.py develop \
            # to enhance the background (non-face) regions and do upscaling
            && pip3 install realesrgan \
            # pre-trained model needed for face restoration
            && wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth -P experiments/pretrained_models 

WORKDIR     /stable-diffusion 

# Preload models 2 GB
RUN         python3 scripts/preload_models.py         

ENTRYPOINT ["/entrypoint.sh"]

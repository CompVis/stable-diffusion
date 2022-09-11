FROM        arm64v8/debian 

ARG         gsd
ENV         GITHUB_STABLE_DIFFUSION $gsd
ARG         sdreq="requirements-linux-arm64.txt"
ENV         SD_REQ $sdreq

WORKDIR     /

COPY        entrypoint.sh anaconda.sh .
SHELL       ["/bin/bash", "-c"]

# Update and apt
RUN         apt update && apt upgrade -y \
            && apt install -y \
            git \
            pip \
            python3 \
            wget 

# install Anaconda or Miniconda
RUN         bash anaconda.sh -b -u -p /anaconda && /anaconda/bin/conda init bash

# SD repo
RUN         git clone $GITHUB_STABLE_DIFFUSION 

WORKDIR     /stable-diffusion 

# SD env
RUN         PIP_EXISTS_ACTION="w" \
            # restrict the Conda environment to only use ARM packages. M1/M2 is ARM-based. You could also conda install nomkl.
            && CONDA_SUBDIR="osx-arm64" \
            # Create the environment, activate it, install requirements.
            && source ~/.bashrc && conda create -y --name ldm && conda activate ldm \
            && pip3 install -r $SD_REQ \                     
            && mkdir models/ldm/stable-diffusion-v1 

# Face restoration prerequisites
RUN         apt install -y libgl1-mesa-glx libglib2.0-0 
            # by default expected in a sibling directory to stable-diffusion

WORKDIR     /       

# Face restoreation repo
RUN         git clone https://github.com/TencentARC/GFPGAN.git 

WORKDIR     /GFPGAN 

# Face restoration env
RUN         pip3 install basicsr facexlib \
            && pip3 install -r requirements.txt \
            && python3 setup.py develop \
            # to enhance the background (non-face) regions and do upscaling
            && pip3 install realesrgan \
            # pre-trained model needed for face restoration
            && wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth -P experiments/pretrained_models 

WORKDIR     /stable-diffusion 

# Preload models
RUN         python3 scripts/preload_models.py         

ENTRYPOINT ["/entrypoint.sh"]

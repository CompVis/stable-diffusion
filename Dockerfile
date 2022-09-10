FROM        arm64v8/debian 
MAINTAINER  Armando C. Santisbon

ARG         gsd
ENV         GITHUB_STABLE_DIFFUSION $gsd

ARG         sdreq="requirements-linux-arm64.txt"
ENV         SD_REQ $sdreq

WORKDIR     /
COPY        entrypoint.sh anaconda.sh .
SHELL       ["/bin/bash", "-c"]

RUN         apt update && apt upgrade -y \
            && apt install -y \
            git \
            pip \
            python3 \
            wget \
            # install Anaconda or Miniconda
            && chmod +x anaconda.sh && bash anaconda.sh -b -u -p /anaconda && /anaconda/bin/conda init bash && source ~/.bashrc \
            && git clone $GITHUB_STABLE_DIFFUSION && cd stable-diffusion \
            # When path exists, pip3 will (w)ipe. 
            && PIP_EXISTS_ACTION="w" \
            # restrict the Conda environment to only use ARM packages. M1/M2 is ARM-based. You could also conda install nomkl.
            && CONDA_SUBDIR="osx-arm64" \
            # Create the environment, activate it, install requirements.
            && conda create -y --name ldm && conda activate ldm \
            && pip3 install -r $SD_REQ \
            
            # Only need to do this once (we'll do it after we add face restoration and upscaling):
            # && python3 scripts/preload_models.py \
            
            && mkdir models/ldm/stable-diffusion-v1 \
            # [Optional] Face Restoration and Upscaling 
            && apt install -y libgl1-mesa-glx libglib2.0-0 \
            # by default expected in a sibling directory to stable-diffusion
            && cd .. && git clone https://github.com/TencentARC/GFPGAN.git && cd GFPGAN \
            && pip3 install basicsr facexlib \
            && pip3 install -r requirements.txt \
            && python3 setup.py develop \
            # to enhance the background (non-face) regions and do upscaling
            && pip3 install realesrgan \
            # pre-trained model needed for face restoration
            && wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth -P experiments/pretrained_models \
            && cd ../stable-diffusion \
            # if we don't preload models it will download model files from the Internet the first time you run dream.py with GFPGAN and Real-ESRGAN turned on.
            && python3 scripts/preload_models.py 

ENTRYPOINT ["/entrypoint.sh"]

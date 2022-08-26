#!/bin/bash
# Starts the gui inside the docker container using the conda env
#

# Activate conda env for this script
# @see https://github.com/ContinuumIO/docker-images/issues/89#issuecomment-467287039
. /opt/conda/etc/profile.d/conda.sh
conda activate ldm

# Validate model files
echo "Validating model files..."

GFPGAN_MODEL="/src/src/gfpgan/experiments/pretrained_models/GFPGANv1.3.pth"
sha256sum --check --status <<< "c953a88f2727c85c3d9ae72e2bd4846bbaf59fe6972ad94130e23e7017524a70 ${GFPGAN_MODEL}"
if [[ $? == "1" ]]; then
    echo "downloading GFPGAN_MODEL please wait..."
    mkdir -p /src/src/gfpgan/experiments/pretrained_models
    wget --output-document=$GFPGAN_MODEL -q --show-progress --progress=bar:force https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth
else
    echo "GFPGAN_MODEL is valid"
fi

SD_MODEL="/src/models/ldm/stable-diffusion-v1/model.ckpt"
sha256sum --check --status <<< "fe4efff1e174c627256e44ec2991ba279b3816e364b49f9be2abc0b3ff3f8556 ${SD_MODEL}"
if [[ $? == "1" ]]; then
    echo "downloading SD_MODEL please wait..."
    mkdir -p /src/models/ldm/stable-diffusion-v1
    wget --output-document=$SD_MODEL -q --show-progress --progress=bar:force https://www.googleapis.com/storage/v1/b/aai-blog-files/o/sd-v1-4.ckpt?alt=media
else
    echo "SD_MODEL is valid"
fi

# Launch web gui
cd /src
python -u scripts/relauncher.py

#!/bin/bash
#
# Starts the gui inside the docker container using the conda env
#

# Array of model files to pre-download
# local filename
# local path in container (no trailing slash)
# download URL
# sha256sum
MODEL_FILES=(
    'model.ckpt /src/models/ldm/stable-diffusion-v1 https://www.googleapis.com/storage/v1/b/aai-blog-files/o/sd-v1-4.ckpt?alt=media fe4efff1e174c627256e44ec2991ba279b3816e364b49f9be2abc0b3ff3f8556'
    'GFPGANv1.3.pth /src/src/gfpgan/experiments/pretrained_models https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth c953a88f2727c85c3d9ae72e2bd4846bbaf59fe6972ad94130e23e7017524a70'
    'RealESRGAN_x4plus.pth /src/src/realesrgan/experiments/pretrained_models https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth 4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1'
    'RealESRGAN_x4plus_anime_6B.pth /src/src/realesrgan/experiments/pretrained_models https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth f872d837d3c90ed2e05227bed711af5671a6fd1c9f7d7e91c911a61f155e99da'
)

# Activate conda env for this script
# @see https://github.com/ContinuumIO/docker-images/issues/89#issuecomment-467287039
cd /src
. /opt/conda/etc/profile.d/conda.sh
conda activate ldm

# Check if environment.yaml file was updated and update env if so
ENV_CREATED_FILE="/tmp/.env_created"
ENV_MODIFED_FILE="/src/.env_updated"
ENV_MODIFIED=$(date -r /src/environment.yaml "+%s")
ENV_CREATED_CACHED=0
ENV_MODIFIED_CACHED=0
if [[ -f $ENV_CREATED_FILE ]]; then ENV_CREATED_CACHED=$(<${ENV_CREATED_FILE}); fi
if [[ -f $ENV_MODIFED_FILE ]]; then ENV_MODIFIED_CACHED=$(<${ENV_MODIFED_FILE}); fi

if (( $ENV_MODIFIED > $ENV_CREATED_CACHED && $ENV_MODIFIED > $ENV_MODIFIED_CACHED )); then
    conda env update --file environment.yaml --prune
    conda clean --all
    echo -n $ENV_MODIFIED > $ENV_MODIFED_FILE
fi

# Function to checks for valid hash for model files and download/replaces if invalid or does not exist
validateDownloadModel() {
    local file=$1
    local path=$2
    local url=$3
    local hash=$4

    echo "checking ${file}..."
    sha256sum --check --status <<< "${hash} ${path}/${file}"
    if [[ $? == "1" ]]; then
        echo "Downloading: ${url} please wait..."
        mkdir -p ${path}
        wget --output-document=${path}/${file} --no-verbose --show-progress --progress=dot:giga ${url}
        echo "saved ${file}"
    else
        echo -e "${file} is valid!\n"
    fi
}

# Validate model files
echo "Validating model files..."
for models in "${MODEL_FILES[@]}"; do
    model=($models)
    validateDownloadModel ${model[0]} ${model[1]} ${model[2]} ${model[3]}
done

# Launch web gui
python -u scripts/relauncher.py

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
    'model.ckpt /sd/models/ldm/stable-diffusion-v1 https://www.googleapis.com/storage/v1/b/aai-blog-files/o/sd-v1-4.ckpt?alt=media fe4efff1e174c627256e44ec2991ba279b3816e364b49f9be2abc0b3ff3f8556'
    'GFPGANv1.3.pth /sd/src/gfpgan/experiments/pretrained_models https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth c953a88f2727c85c3d9ae72e2bd4846bbaf59fe6972ad94130e23e7017524a70'
    'RealESRGAN_x4plus.pth /sd/src/realesrgan/experiments/pretrained_models https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth 4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1'
    'RealESRGAN_x4plus_anime_6B.pth /sd/src/realesrgan/experiments/pretrained_models https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth f872d837d3c90ed2e05227bed711af5671a6fd1c9f7d7e91c911a61f155e99da'
)

# Conda environment installs/updates
# @see https://github.com/ContinuumIO/docker-images/issues/89#issuecomment-467287039
ENV_NAME="ldm"
ENV_FILE="/sd/environment.yaml"
ENV_UPDATED=0
ENV_MODIFIED=$(date -r $ENV_FILE "+%s")
ENV_MODIFED_FILE="/sd/.env_updated"
if [[ -f $ENV_MODIFED_FILE ]]; then ENV_MODIFIED_CACHED=$(<${ENV_MODIFED_FILE}); else ENV_MODIFIED_CACHED=0; fi

# Create/update conda env if needed
if ! conda env list | grep ".*${ENV_NAME}.*" >/dev/null 2>&1; then
    echo "Could not find conda env: ${ENV_NAME} ... creating ..."
    conda env create -f $ENV_FILE
    echo "source activate ${ENV_NAME}" > /root/.bashrc
    ENV_UPDATED=1
elif [[ ! -z $CONDA_FORCE_UPDATE && $CONDA_FORCE_UPDATE == "true" ]] || (( $ENV_MODIFIED > $ENV_MODIFIED_CACHED )); then
    echo "Updating conda env: ${ENV_NAME} ..."
    conda env update --file $ENV_FILE --prune
    ENV_UPDATED=1
fi

# Clear artifacts from conda after create/update
# @see https://docs.conda.io/projects/conda/en/latest/commands/clean.html
if (( $ENV_UPDATED > 0 )); then
    conda clean --all
    echo -n $ENV_MODIFIED > $ENV_MODIFED_FILE
fi

# activate conda env
. /opt/conda/etc/profile.d/conda.sh
conda activate $ENV_NAME
conda info | grep active

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
if [[ -z $VALIDATE_MODELS || $VALIDATE_MODELS == "true" ]]; then
    echo "Validating model files..."
    for models in "${MODEL_FILES[@]}"; do
        model=($models)
        validateDownloadModel ${model[0]} ${model[1]} ${model[2]} ${model[3]}
    done
fi

# Launch web gui
cd /sd

if [[ -z $WEBUI_ARGS ]]; then
    launch_message="entrypoint.sh: Launching..."
else
    launch_message="entrypoint.sh: Launching with arguments ${WEBUI_ARGS}"
fi

if [[ -z $WEBUI_RELAUNCH || $WEBUI_RELAUNCH == "true" ]]; then
    n=0
    while true; do

        echo $launch_message
        if (( $n > 0 )); then
            echo "Relaunch count: ${n}"
        fi
        python -u scripts/webui.py $WEBUI_ARGS
        echo "entrypoint.sh: Process is ending. Relaunching in 0.5s..."
        ((n++))
        sleep 0.5
    done
else
    echo $launch_message
    python -u scripts/webui.py $WEBUI_ARGS
fi

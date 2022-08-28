#!/bin/bash
# It basically resets you to the beginning except for your output directory.
# How to:
#     cd stable-diffusion
#     ./docker-reset.sh
# Then:
#     docker-compose up

echo $(pwd)
read -p "Is the directory above correct to run reset on? (y/n) " -n 1 DIRCONFIRM
if [[ $DIRCONFIRM =~ ^[Yy]$ ]]; then
    docker compose down
    docker image rm stable-diffusion_stable-diffusion:latest
    docker volume rm stable-diffusion_conda_env
    docker volume rm stable-diffusion_root_profile
    echo "Remove ./src"
    sudo rm -rf src
    sudo rm -rf latent_diffusion.egg-info
    sudo rm .env_updated
else
    echo "Exited without resetting"
fi

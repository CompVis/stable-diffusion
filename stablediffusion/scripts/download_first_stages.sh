#!/bin/bash
wget -O models/first_stage_models/kl-f4/model.zip https://ommer-lab.com/files/latent-diffusion/kl-f4.zip
wget -O models/first_stage_models/kl-f8/model.zip https://ommer-lab.com/files/latent-diffusion/kl-f8.zip
wget -O models/first_stage_models/kl-f16/model.zip https://ommer-lab.com/files/latent-diffusion/kl-f16.zip
wget -O models/first_stage_models/kl-f32/model.zip https://ommer-lab.com/files/latent-diffusion/kl-f32.zip
wget -O models/first_stage_models/vq-f4/model.zip https://ommer-lab.com/files/latent-diffusion/vq-f4.zip
wget -O models/first_stage_models/vq-f4-noattn/model.zip https://ommer-lab.com/files/latent-diffusion/vq-f4-noattn.zip
wget -O models/first_stage_models/vq-f8/model.zip https://ommer-lab.com/files/latent-diffusion/vq-f8.zip
wget -O models/first_stage_models/vq-f8-n256/model.zip https://ommer-lab.com/files/latent-diffusion/vq-f8-n256.zip
wget -O models/first_stage_models/vq-f16/model.zip https://ommer-lab.com/files/latent-diffusion/vq-f16.zip



cd models/first_stage_models/kl-f4
unzip -o model.zip

cd ../kl-f8
unzip -o model.zip

cd ../kl-f16
unzip -o model.zip

cd ../kl-f32
unzip -o model.zip

cd ../vq-f4
unzip -o model.zip

cd ../vq-f4-noattn
unzip -o model.zip

cd ../vq-f8
unzip -o model.zip

cd ../vq-f8-n256
unzip -o model.zip

cd ../vq-f16
unzip -o model.zip

cd ../..
FROM continuumio/miniconda3:4.12.0 AS build

# Step for image utility dependencies.
RUN apt update \
 && apt install --no-install-recommends -y git \
 && apt-get clean

COPY . /root/stable-diffusion/

# Step to install dependencies with conda
RUN eval "$(conda shell.bash hook)" \
 && conda install -c conda-forge conda-pack \
 && conda env create -f /root/stable-diffusion/environment.yaml \
 && conda activate ldm \
 && pip install gradio==3.1.7 \
 && conda activate base  

# Step to zip and conda environment to "venv" folder
RUN conda pack --ignore-missing-files --ignore-editable-packages -n ldm -o /tmp/env.tar \
 && mkdir /venv \
 && cd /venv \
 && tar xf /tmp/env.tar \
 && rm /tmp/env.tar

FROM nvidia/cuda:11.8.0-base-ubuntu22.04 as runtime

ARG OPTIMIZED_FILE=txt2img_gradio.py
WORKDIR /root/stable-diffusion

COPY --from=build /venv /venv
COPY --from=build /root/stable-diffusion /root/stable-diffusion

RUN mkdir -p /output /root/stable-diffusion/outputs \
 && ln -s /data /root/stable-diffusion/models/ldm/stable-diffusion-v1 \
 && ln -s /output /root/stable-diffusion/outputs/txt2img-samples

ENV PYTHONUNBUFFERED=1
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860 
ENV APP_MAIN_FILE=${OPTIMIZED_FILE}
EXPOSE 7860

VOLUME ["/root/.cache", "/data", "/output"]

SHELL ["/bin/bash", "-c"]
ENTRYPOINT ["/root/stable-diffusion/docker-bootstrap.sh"]
CMD python optimizedSD/${APP_MAIN_FILE}
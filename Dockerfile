FROM nvcr.io/nvidia/pytorch:22.10-py3

WORKDIR /research

ENV HOME /research

RUN pip install omegaconf==2.1.1 diffusers invisible-watermark opencv-python-headless==4.1.2.30 einops==0.3.0 pytorch-lightning==1.4.2  torchmetrics==0.6.0 kornia==0.6 transformers==4.19.2

# Mount data into the docker
ADD . /research/stable-diffusion


WORKDIR /research/stable-diffusion
ENV PYTHONPATH /research/stable-diffusion

RUN mkdir src && cd src && \
    git clone https://github.com/CompVis/taming-transformers.git && cd taming-transformers && \
    pip install -e . && cd ../
RUN git clone https://github.com/openai/CLIP.git && cd CLIP && \
    pip install -e . && cd ../../

ENTRYPOINT ["/bin/bash"]


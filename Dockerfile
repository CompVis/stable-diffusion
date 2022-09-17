FROM pytorch/pytorch:latest

RUN apt update && \
    apt install -y git curl unzip vim && \
    pip install git+https://github.com/derfred/lightning.git@waifu-1.6.0#egg=pytorch-lightning
RUN mkdir /waifu
COPY . /waifu/
WORKDIR /waifu
RUN grep -v pytorch-lightning requirements.txt > requirements-waifu.txt && \
    pip install -r requirements-waifu.txt

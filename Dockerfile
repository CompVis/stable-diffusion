FROM pytorch/pytorch:latest

RUN mkdir /waifu
COPY . /waifu/
WORKDIR /waifu
RUN pip install -r requirement.txt

FROM ubuntu:22.04

ENV CONDA_SRC=/conda

# Install conda
USER root

ENV TZ=Europe
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN mkdir -p ${CONDA_SRC} && \ 
    apt-get update && \ 
    apt-get install -y wget && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ${CONDA_SRC}/miniconda.sh && \
    bash ${CONDA_SRC}/miniconda.sh -b -u -p ${CONDA_SRC} && \
    rm -rf ${CONDA_SRC}/miniconda.sh  && \
    ${CONDA_SRC}/bin/conda init bash  && \
    ${CONDA_SRC}/bin/conda init zsh && \
    ${CONDA_SRC}/bin/conda clean -ay

RUN apt-get update && apt-get install -y git-all

# Create the environment:
COPY environment.yaml setup.py .
     
RUN ${CONDA_SRC}/bin/conda env create -f environment.yaml
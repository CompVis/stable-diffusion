FROM nvidia/cuda:11.3.1-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash", "-c"]

RUN apt update && \
    apt install -y libglib2.0-0 wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install miniconda
ENV CONDA_DIR /opt/conda
RUN wget -O ~/miniconda.sh -q --show-progress --progress=bar:force https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    /bin/bash ~/miniconda.sh -b -p $CONDA_DIR && \
    rm ~/miniconda.sh
ENV PATH=$CONDA_DIR/bin:$PATH

COPY ./environment.yaml ./setup.py /src/
WORKDIR /src

# setup env
RUN conda env create -f /src/environment.yaml && \
    echo "source activate ldm" > /root/.bashrc && \
    conda clean --all && \
    echo -n $(date "+%s") > /tmp/.env_created

EXPOSE 7860

COPY ./entrypoint.sh /src/
ENTRYPOINT /src/entrypoint.sh

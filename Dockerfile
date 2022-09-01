FROM nvidia/cuda:11.3.1-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /sd

SHELL ["/bin/bash", "-c"]

RUN apt-get update && \
    apt-get install -y libglib2.0-0 wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install miniconda
ENV CONDA_DIR /opt/conda
RUN wget -O ~/miniconda.sh -q --show-progress --progress=bar:force https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    /bin/bash ~/miniconda.sh -b -p $CONDA_DIR && \
    rm ~/miniconda.sh
ENV PATH=$CONDA_DIR/bin:$PATH

# Install font for prompt matrix
COPY /data/DejaVuSans.ttf /usr/share/fonts/truetype/

EXPOSE 7860

COPY    entrypoint.sh /sd/
COPY    entrypoint_docker.sh /sd/
COPY 	assets /sd/assets/   
COPY 	configs /sd/configs/ 
COPY 	data /sd/data/   
COPY 	docker-compose.yml /sd/   
COPY 	Dockerfile /sd/   
COPY 	docker-reset.sh /sd/   
COPY 	environment.yaml /sd/   
COPY 	environment.yml /sd/   
COPY 	frontend /sd/frontend/   
COPY 	install.sh /sd/
COPY 	ldm /sd/ldm/   
COPY 	LICENSE /sd/   
COPY 	main.py /sd/   
COPY 	models /sd/models/   
COPY 	notebook_helpers.py /sd/   
COPY 	optimizedSD /sd/optimizedSD/   
COPY 	README.md /sd/   
COPY 	scripts /sd/scripts   
COPY 	setup.py /sd/   
COPY 	src /sd/src/   
COPY 	Stable_Diffusion_v1_Model_Card.md /sd/   
COPY 	txt2img.yaml /sd/   
COPY 	webui.cmd /sd/   
COPY 	webuildm.cmd /sd/   
ENTRYPOINT /sd/entrypoint_docker.sh

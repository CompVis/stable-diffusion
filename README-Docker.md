
Table of Contents
=================

* [Step 1 - Get the Model](#step-1---get-the-model)
* [Step 2 - Installation](#step-2---installation)
   * [Option A - On a Linux container](#option-a---on-a-linux-container)
      * [Why containers?](#why-containers)
      * [Prerequisites](#prerequisites)
      * [Setup](#setup)
   * [Option B - Directly on Apple silicon](#option-b---directly-on-apple-silicon)
      * [Prerequisites](#prerequisites-1)
      * [Setup](#setup-1)
* [Step 3 - Usage (time to have fun)](#step-3---usage-time-to-have-fun)
   * [Startup](#startup)
   * [Text to Image](#text-to-image)
   * [Image to Image](#image-to-image)
   * [Web Interface](#web-interface)
   * [Notes](#notes)


# Step 1 - Get the Model
Go to [Hugging Face](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original), and click "Access repository" to Download ```sd-v1-4.ckpt``` (~4 GB) to ```~/Downloads```.  
You'll need to create an account but it's quick and free.

# Step 2 - Installation

## Option A - On a Linux container 

### Why containers?
They provide a flexible, reliable way to build and deploy Stable Diffusion. We also use a Docker volume to store the largest model file and image outputs as a first step in decoupling storage and compute. Future enhancements will do this for other model files and assets. See [Processes](https://12factor.net/processes) under the Twelve-Factor App methodology for details on why running applications in such a stateless fashion is important.

This example uses a Mac M1/M2 (arm64) but you can specify the platform and architecture as parameters when building the image and running the container.  

The steps would be the same on an amd64 machine with NVIDIA GPUs as for an arm64 Mac; the platform is configurable. You [can't access the Mac M1/M2 GPU cores from Docker containers](https://github.com/pytorch/pytorch/issues/81224) and performance is reduced compared with running it directly on macOS but for development purposes it's fine. Once you're done with development tasks on your laptop you can build for the target platform and architecture and deploy to an environment with NVIDIA GPUs on-premises or in the cloud.

### Prerequisites
[Install Docker](https://gist.github.com/santisbon/2165fd1c9aaa1f7974f424535d3756f7#docker)  
On the Docker Desktop app, go to Preferences, Resources, Advanced. Increase the CPUs and Memory to avoid this [Issue](https://github.com/lstein/stable-diffusion/issues/342). You may need to increase Swap and Disk image size too.  

Create a Docker volume for the downloaded model file
```
docker volume create my-vol
```

Populate the volume using a lightweight Linux container. You just need to create the container with the mountpoint; no need to run it.
```Shell
docker create --platform linux/arm64 --name dummy --mount source=my-vol,target=/data alpine # or arm64v8/alpine

# Copy the model file to the Docker volume. We'll need it at run time.
cd ~/Downloads # or wherever you saved sd-v1-4.ckpt
docker cp sd-v1-4.ckpt dummy:/data
```

### Setup
```Shell
# Set the fork you want to use.
GITHUB_STABLE_DIFFUSION="https://github.com/santisbon/stable-diffusion.git"

cd ~
git clone $GITHUB_STABLE_DIFFUSION

cd stable-diffusion/docker-build
chmod +x entrypoint.sh
# Download the Miniconda installer. We'll need it at build time. 
# Replace the URL with the version matching your system.
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh -O anaconda.sh && chmod +x anaconda.sh
```

Build the Docker image. Give it any tag ```-t``` that you want.  
Tip: Check that your shell session has the env variable set (above) with ```echo $GITHUB_STABLE_DIFFUSION```.  
```condaarch``` will restrict the conda environment to the right architecture when installing packages. It can take on: ```linux-64```, ```osx-64```, ```osx-arm64```. On macOS you could also conda install ```nomkl``` but setting the environment appropriately is cleaner.
```Shell
docker build -t santisbon/stable-diffusion \
--platform linux/arm64 \
--build-arg condaarch="osx-arm64" \
--build-arg gsd=$GITHUB_STABLE_DIFFUSION \
--build-arg sdreq="requirements-linux-arm64.txt" \
.
```

Run a container using your built image e.g.
```Shell
docker run -it \
--rm \
--platform linux/arm64 \
--name stable-diffusion \
--hostname stable-diffusion \
--mount source=my-vol,target=/data \
--expose 9090 \
--publish 9090:9090 \
santisbon/stable-diffusion
```
Tip: Make sure you've created the Docker volume (above)

## Option B - Directly on Apple silicon
For Mac M1/M2. Read more about [Metal Performance Shaders (MPS) framework](https://developer.apple.com/documentation/metalperformanceshaders).

### Prerequisites
Install the latest versions of macOS, [Homebrew](https://brew.sh/), [Python](https://gist.github.com/santisbon/2165fd1c9aaa1f7974f424535d3756f7#python), and [Git](https://gist.github.com/santisbon/2165fd1c9aaa1f7974f424535d3756f7#git).  

```Shell
brew install cmake protobuf rust
brew install --cask miniconda
conda init zsh && source ~/.zshrc # or bash and .bashrc
```

### Setup

```Shell
# Set the fork you want to use.
GITHUB_STABLE_DIFFUSION="https://github.com/santisbon/stable-diffusion.git"

git clone $GITHUB_STABLE_DIFFUSION
cd stable-diffusion
mkdir -p models/ldm/stable-diffusion-v1/
```

```Shell
PATH_TO_CKPT="$HOME/Downloads"  # or wherever you saved sd-v1-4.ckpt
ln -s "$PATH_TO_CKPT/sd-v1-4.ckpt" models/ldm/stable-diffusion-v1/model.ckpt

# When path exists, pip3 will (w)ipe. 
# restrict the Conda environment to only use ARM packages. M1/M2 is ARM-based. You could also conda install nomkl.
PIP_EXISTS_ACTION="w"
CONDA_SUBDIR="osx-arm64"
conda env create -f environment-mac.yaml && conda activate ldm
```

You can verify you're in the virtual environment by looking at which executable you're getting:
```Shell
type python3
```

Face Restoration and Upscaling 
```Shell

# by default expected in a sibling directory to stable-diffusion
cd .. && git clone https://github.com/TencentARC/GFPGAN.git && cd GFPGAN

# basicsr: used for training and inference. facexlib: face detection / face restoration helper.
pip3 install basicsr facexlib \
&& pip3 install -r requirements.txt

python3 setup.py develop
pip3 install realesrgan # to enhance the background (non-face) regions and do upscaling
# pre-trained model needed for face restoration
wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth -P experiments/pretrained_models

cd ../stable-diffusion
```

Only need to do this once. If we don't preload models it will download model files from the Internet when you run ```dream.py```. Used by the core functionality and by GFPGAN/Real-ESRGAN.
```Shell
python3 scripts/preload_models.py
```

# Step 3 - Usage (time to have fun)

## Startup
If you're on a Linux container the ```dream``` script is automatically started and the output dir set to the Docker volume you created earlier. 

If you're directly on macOS follow these startup instructions.  
With the Conda environment activated (```conda activate ldm```), run the interactive interface that combines the functionality of the original scripts txt2img and img2img:
Use the more accurate but VRAM-intensive full precision math because half-precision requires autocast and won't work.

```Shell
# If on Macbook
python3 scripts/dream.py --full_precision
# By default the images are saved in outputs/img-samples/.  
```

You'll get the script's prompt. You can see available options or quit.
```Shell
dream> -h
dream> q
```

## Text to Image
For quick (but bad) image results test with 5 steps (default 50) and 1 sample image. This will let you know that everything is set up correctly.  
Then increase steps to 100 or more for good (but slower) results.  
The prompt can be in quotes or not.
```Shell
dream> The hulk fighting with sheldon cooper -s5 -n1 
dream> "woman closeup highly detailed"  -s 150
# Reuse previous seed and apply face restoration (if you installed GFPGAN)
dream> "woman closeup highly detailed"  --steps 150 --seed -1 -G 0.75
# TODO: example for upscaling.
```
You'll need to experiment to see if face restoration is making it better or worse for your specific prompt.
The -U option for upscaling has an [Issue](https://github.com/lstein/stable-diffusion/issues/297).  

If you're on a container the output is set to the Docker volume. You can copy it wherever you want.  
You can download it from the Docker Desktop app, Volumes, my-vol, data.  
Or you can copy it from your Mac terminal. Keep in mind ```docker cp``` can't expand ```*.png``` so you'll need to specify the image file name:
```Shell
# On your host Macbook (you can use the name of any container that mounted the volume)
docker cp dummy:/data/000001.928403745.png /Users/<your-user>/Pictures 
```

## Image to Image
You can also do text-guided image-to-image translation. For example, turning a sketch into a detailed drawing.  
Strength is a value between 0.0 and 1.0, that controls the amount of noise that is added to the input image. Values that approach 1.0 allow for lots of variations but will also produce images that are not semantically consistent with the input. 0.0 preserves image exactly, 1.0 replaces it completely.  
Make sure your input image size dimensions are multiples of 64 e.g. 512x512. Otherwise you'll get ```Error: product of dimension sizes > 2**31'```. If you still get the error [try a different size](https://support.apple.com/guide/preview/resize-rotate-or-flip-an-image-prvw2015/mac#:~:text=image's%20file%20size-,In%20the%20Preview%20app%20on%20your%20Mac%2C%20open%20the%20file,is%20shown%20at%20the%20bottom.) like 512x256.  

If you're on a docker container, copy your input image into the Docker volume
```Shell
docker cp /Users/<your-user>/Pictures/sketch-mountains-input.jpg dummy:/data/
```

Try it out generating an image (or 4).  
The ```dream``` script needs absolute paths to find the image so don't use ```~```.
```Shell
# If you're on your Macbook 
dream> "A fantasy landscape, trending on artstation" -I /Users/<your-user>/Pictures/sketch-mountains-input.jpg --strength 0.8  --steps 100 -n4
# If you're on a Linux container on your Macbook
dream> "A fantasy landscape, trending on artstation" -I /data/sketch-mountains-input.jpg --strength 0.75  --steps 100 -n1
```

## Web Interface
You can use the ```dream``` script with a graphical web interface. Start the web server with:
```Shell
python3 scripts/dream.py --full_precision --web
```
If it's running on your Mac point your Mac web browser to http://127.0.0.1:9090  

Press Control-C at the command line to stop the web server.

## Notes

Some text you can add at the end of the prompt to make it very pretty:
```Shell
cinematic photo, highly detailed, cinematic lighting, ultra-detailed, ultrarealistic, photorealism, Octane Rendering, cyberpunk lights, Hyper Detail, 8K, HD, Unreal Engine, V-Ray, full hd, cyberpunk, abstract, 3d octane render + 4k UHD + immense detail + dramatic lighting + well lit + black, purple, blue, pink, cerulean, teal, metallic colours, + fine details, ultra photoreal, photographic, concept art, cinematic composition, rule of thirds, mysterious, eerie, photorealism, breathtaking detailed, painting art deco pattern, by hsiao, ron cheng, john james audubon, bizarre compositions, exquisite detail, extremely moody lighting, painted by greg rutkowski makoto shinkai takashi takeuchi studio ghibli, akihiko yoshida
```

The original scripts should work as well.
```Shell
python3 scripts/orig_scripts/txt2img.py --help
python3 scripts/orig_scripts/txt2img.py --ddim_steps 100 --n_iter 1 --n_samples 1  --plms --prompt "new born baby kitten. Hyper Detail, Octane Rendering, Unreal Engine, V-Ray"
python3 scripts/orig_scripts/txt2img.py --ddim_steps 5   --n_iter 1 --n_samples 1  --plms --prompt "ocean" # or --klms
```


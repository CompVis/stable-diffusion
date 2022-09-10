
Table of Contents
=================

* [Installation](#installation)
   * [Option 1 - Directly on Apple silicon](#option-1---directly-on-apple-silicon)
      * [Prerequisites](#prerequisites)
      * [Setup](#setup)
   * [Option 2 - On a Linux container with Docker for Apple silicon](#option-2---on-a-linux-container-with-docker-for-apple-silicon)
      * [Prerequisites](#prerequisites-1)
      * [Setup](#setup-1)
   * [[Optional] Face Restoration and Upscaling](#optional-face-restoration-and-upscaling)
      * [Setup](#setup-2)
* [Usage](#usage)
   * [Startup](#startup)
   * [Text to Image](#text-to-image)
   * [Image to Image](#image-to-image)
   * [Web Interface](#web-interface)
   * [Notes](#notes)


Go to [Hugging Face](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original), and click "Access repository" to Download ```sd-v1-4.ckpt``` (~4 GB) to ```~/Downloads```.  
You'll need to create an account but it's quick and free.

# Installation

## Option 1 - Directly on Apple silicon
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
GITHUB_STABLE_DIFFUSION=https://github.com/santisbon/stable-diffusion.git

git clone $GITHUB_STABLE_DIFFUSION
cd stable-diffusion
mkdir -p models/ldm/stable-diffusion-v1/
```

```Shell
PATH_TO_CKPT="$HOME/Downloads"  # or wherever you saved sd-v1-4.ckpt
ln -s "$PATH_TO_CKPT/sd-v1-4.ckpt" models/ldm/stable-diffusion-v1/model.ckpt

# When path exists, pip3 will (w)ipe. 
# restrict the Conda environment to only use ARM packages. M1/M2 is ARM-based. You could also conda install nomkl.
PIP_EXISTS_ACTION=w
CONDA_SUBDIR=osx-arm64
conda env create -f environment-mac.yaml && conda activate ldm
```

You can verify you're in the virtual environment by looking at which executable you're getting:
```Shell
type python3
```

Only need to do this once:
```Shell
python3 scripts/preload_models.py
```

## Option 2 - On a Linux container with Docker for Apple silicon
You [can't access the Macbook M1/M2 GPU cores from the Docker containers](https://github.com/pytorch/pytorch/issues/81224) so performance is reduced but for development purposes it's fine.

### Prerequisites
[Install Docker](https://gist.github.com/santisbon/2165fd1c9aaa1f7974f424535d3756f7#install-2)  
On the Docker Desktop app, go to Preferences, Resources, Advanced. Increase the CPUs and Memory to avoid this [Issue](https://github.com/lstein/stable-diffusion/issues/342). You may need to increase Swap and Disk image size too.  

Create a Docker volume for the downloaded model file
```
docker volume create my-vol
```

Populate the volume using a lightweight Linux container. You just need to create the container with the mountpoint; no need to run it.
```Shell
docker create --platform linux/arm64 --name dummy --mount source=my-vol,target=/data alpine # or arm64v8/alpine
cd ~/Downloads # or wherever you saved sd-v1-4.ckpt
docker cp sd-v1-4.ckpt dummy:/data
```

### Setup
Start a container for Stable Diffusion. The container's 9090 port is mapped to the host's 80. That way you'll be able to use the Web interface from your Mac.
```Shell
docker run -it \
--platform linux/arm64 \
--name stable-diffusion \
--hostname stable-diffusion \
--mount source=my-vol,target=/data \
--expose 9090 \
--publish 80:9090 \
debian
# or arm64v8/debian
```

You're now on the container.
```Shell
# Set the fork you want to use
GITHUB_STABLE_DIFFUSION="-b docker-apple-silicon https://github.com/santisbon/stable-diffusion.git" \
&& apt update && apt upgrade -y \
&& apt install -y \
git \
pip \
python3 \
wget

# you won't need to close and reopen your terminal after this because we'll source our .<shell>rc file
cd /data && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh -O anaconda.sh \
&& chmod +x anaconda.sh && bash anaconda.sh -b -u -p /anaconda && /anaconda/bin/conda init bash && source ~/.bashrc

cd / && git clone $GITHUB_STABLE_DIFFUSION && cd stable-diffusion 

# When path exists, pip3 will (w)ipe. 
# restrict the Conda environment to only use ARM packages. M1/M2 is ARM-based. You could also conda install nomkl.
PIP_EXISTS_ACTION=w 
CONDA_SUBDIR=osx-arm64 

# Create the environment, activate it, install requirements.
conda create -y --name ldm && conda activate ldm \
&& pip3 install -r requirements-linux-arm64.txt 

# Only need to do this once (ok twice if you decide to add face restoration and upscaling):
python3 scripts/preload_models.py

mkdir -p models/ldm/stable-diffusion-v1 \
&& chown root:root /data/sd-v1-4.ckpt \
&& ln -sf /data/sd-v1-4.ckpt models/ldm/stable-diffusion-v1/model.ckpt
```

## [Optional] Face Restoration and Upscaling 
Whether you're directly on macOS or a Linux container.

### Setup
```Shell
# If you're on a Linux container
apt install -y libgl1-mesa-glx libglib2.0-0

# by default expected in a sibling directory to stable-diffusion
cd .. && git clone https://github.com/TencentARC/GFPGAN.git && cd GFPGAN

# basicsr: used for training and inference. facexlib: face detection / face restoration helper.
pip3 install basicsr facexlib 
pip3 install -r requirements.txt

python3 setup.py develop
pip3 install realesrgan # to enhance the background (non-face) regions and do upscaling
# pre-trained model needed for face restoration
wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth -P experiments/pretrained_models

cd ../stable-diffusion
# if we don't preload models it will download model files from the Internet the first time you run dream.py with GFPGAN and Real-ESRGAN turned on.
python3 scripts/preload_models.py 
```


# Usage

## Startup
With the Conda environment activated (```conda activate ldm```), run the interactive interface that combines the functionality of the original scripts txt2img and img2img:
Use the more accurate but VRAM-intensive full precision math because half-precision requires autocast and won't work.

By default the images are saved in ```outputs/img-samples/```.  
If you're on a docker container set the output dir to the Docker volume you created. 
```Shell
# If on Macbook
python3 scripts/dream.py --full_precision
# If on Linux container
python3 scripts/dream.py --full_precision -o /data 
```

You'll get the script's prompt. You can see available options or quit.
```Shell
dream> -h
dream> q
```

## Text to Image
For quick (but very rough) results test with 5 steps (default 50) and 1 sample image. This will let you know that everything is set up correctly.  
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
The -U option for upscaling has an [Issue](https://github.com/lstein/stable-diffusion/issues/297) on Mac.  

If you're on a container and set the output to the Docker volume (or moved it there with ```mv outputs/img-samples/ /data/```) you can copy it wherever you want.  
You can download it from the Docker Desktop app, Volumes, my-vol, data.  
Or you can copy it from your terminal. Keep in mind ```docker cp``` can't expand ```*.png``` so you'll need to specify the image file name:
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


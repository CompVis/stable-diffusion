# Prog Rock Stable
An enhanced (hopefully!) version of Stable Diffusion

Please consider supporting my time and effort in maintaining and improving this program on my [Patreon](https://www.patreon.com/jasonmhough?fan_landing=true). Thanks!

Also available:
- [Prog Rock Diffusion](https://github.com/lowfuel/progrockdiffusion) (command line Disco Diffusion, with Go_Big and other enhancements)
- [Disco Diffusion notebook](https://github.com/lowfuel/DiscoDiffusion-Warp-gobig) with Go_Big

# Quick updating note:
If you installed before 8/24/2022 and have just grabbed the latest code, you'll need to manually install a few extra items.
Make sure you're in your prs conda environment, then do:
```
pip install jsonmerge clean-fid resize-right torchdiffeq
```

# Installation instructions

## Download Prog Rock Stable
Download this repository either by zip file (click the "Code" option above and select "Download ZIP"), or via git:
```
git clone https://github.com/lowfuel/progrock-stable prs
cd prs
```

## Setup a Conda environment
*(MacOS M1/M2 users, see [here](#macos-setup) for Conda instructions, then move on to the next section)*

Create a [conda](https://conda.io/) environment named `prs`:
```
conda env create -f environment.yaml
conda activate prs
```

## Download Stable Diffusion Weights
Download the [Stable Diffusion weights](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original) "sd-v1-4.ckpt", and place it in the `models` directory

Run prs to make sure everything worked!
```
python prs.py
```

# Basic Use

To use the default settings, but with your own text prompt:
```
python prs.py -p "A painting of a troll under a bridge, by Hubert Robert"
```

# Intermediate Use

It is recommended that you create your own settings file(s) inside the settings folder, and leave the orignial settings.json file as is.

To specify your own settings file, simply do:
```
python prs.py -s settings\my_file.json
```
Note: You can supply multiple settings partial settings files, they will be layered on top of the previous ones in order, ALWAYS starting with the default settings.json. 

# Advanced Use
## Run a series of prompts
Create a text file (let's call it myprompts.txt), then edit your settings file and set:
```
    "from_file": "myprompts.txt",
```
Each prompt will be run, in order, n_batches of times. So if n_batches = 5 you'll get 5 images for the first prompt, then five for the second, and so on.

# GoBIG! What it is and how to use it
GoBIG is an upscaling technique, where a starting image is cut up into sections, and then each of those sections is re-rendered at a higher resolution. Once each section is done, they're all gathered and composited together, resulting in a new image that is 2x the size of the original.

## Use
The simplest method is to add --gobig to your command line. This will render your initial image, then proceed immediately to the gobig process discussed above.

However, more often than not you'll probably want to choose an existing render (or any image really) to start with. To do that, you add --gobig_init to your command *as well*.
```
python prs.py --gobig --gobig_init "init_images/myfile.png"
```
## Fine-tuning GoBIG
There are a few settings you can tweak to improve your results:
- First and foremost is init_strength. This setting determines how much of the original image should be retained, and thus how many steps to skip in the render process. I recommend a number between 0.55 and 0.75, and you will need to experiment to find the perfect setting for your image.

- The second is to use [RealESRGAN](https://github.com/xinntao/Real-ESRGAN/releases) to handle the initial resizing the starting image. To do this, install RealESRGAN and make sure it is in your path, then set "gobig_realesrgan" to "true" in your settings. This will begin your process with a much cleaner image.

- Lastly, consider tweaking the prompt from your original image to one that focuses more on texture and detail. Keep in mind that each section of the image will use the prompt, so if the image you are upscaling has a singular subject in one area (say, a bird), as it re-renders each section if "bird" is in the prompt it may try to add a bird to those smaller sections, resulting in an upscaled image that is not what you wanted.

- Not every image does well with GoBIG. It is best used on images that have lots of content and fine detail everywhere. So, don't force it! Sometimes a simple upscaler like RealESRGAN will do a better job, especially on those images where your prompt might not apply to every section.

- Finally, remember that the output itself doesn't need to be "final". Take the results from GoBIG and load it into an image editor, along with the original and the ESRGAN upscaled version, layer them, and keep the best areas from each for a true final image.

# MacOS Setup

Install Homebrew:
```
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```
Restart your terminal, then:
```
brew install miniforge
conda init zsh
```
Restart your terminal again, and setup your Conda environment:
```
conda env create -f mac-environment.yaml
conda activate prs
```
You can now continue with [installation above](#download-stable-diffusion-weights).

# About Stable Diffusion
*Stable Diffusion was made possible thanks to a collaboration with [Stability AI](https://stability.ai/) and [Runway](https://runwayml.com/) and builds upon our previous work:*

[**High-Resolution Image Synthesis with Latent Diffusion Models**](https://ommer-lab.com/research/latent-diffusion-models/)<br/>
[Robin Rombach](https://github.com/rromb)\*,
[Andreas Blattmann](https://github.com/ablattmann)\*,
[Dominik Lorenz](https://github.com/qp-qp)\,
[Patrick Esser](https://github.com/pesser),
[Björn Ommer](https://hci.iwr.uni-heidelberg.de/Staff/bommer)<br/>


# Prog Rock Stable
An enhanced (hopefully!) version of Stable Diffusion

Please consider supporting my time and effort in maintaining and improving this program on my [Patreon](https://www.patreon.com/jasonmhough?fan_landing=true). Thanks!

Also available:
- [Prog Rock Diffusion](https://github.com/lowfuel/progrockdiffusion) (command line Disco Diffusion, with Go_Big and other enhancements)
- [Disco Diffusion notebook](https://github.com/lowfuel/DiscoDiffusion-Warp-gobig) with Go_Big

# Installation instructions
Download this repository either by zip file or via git:
```
git clone --no-checkout https://github.com/lowfuel/progrock-stable prs
cd prs
```

Create a [conda](https://conda.io/) environment named `prs`:
```
conda env create -f environment.yaml
conda activate prs
```

Download the model required for Stable Diffusion, and copy it into the `models` directory

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

# About Stable Diffusion
*Stable Diffusion was made possible thanks to a collaboration with [Stability AI](https://stability.ai/) and [Runway](https://runwayml.com/) and builds upon our previous work:*

[**High-Resolution Image Synthesis with Latent Diffusion Models**](https://ommer-lab.com/research/latent-diffusion-models/)<br/>
[Robin Rombach](https://github.com/rromb)\*,
[Andreas Blattmann](https://github.com/ablattmann)\*,
[Dominik Lorenz](https://github.com/qp-qp)\,
[Patrick Esser](https://github.com/pesser),
[Björn Ommer](https://hci.iwr.uni-heidelberg.de/Staff/bommer)<br/>


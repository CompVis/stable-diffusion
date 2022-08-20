# Prog Rock Stable
An enhanced (hopefully!) version of Stable Diffusion

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

# Use
It is recommended that you create your own settings file(s) inside the settings folder, and leave the orignial settings.json file as is.

To specify your own settings file, simply do:
```
python prs.py -s settings\my_file.json
```
Note: You can supply multiple settings partial settings files, they will be layered on top of the previous ones in order, ALWAYS starting with the default settings.json


# About Stable Diffusion
*Stable Diffusion was made possible thanks to a collaboration with [Stability AI](https://stability.ai/) and [Runway](https://runwayml.com/) and builds upon our previous work:*

[**High-Resolution Image Synthesis with Latent Diffusion Models**](https://ommer-lab.com/research/latent-diffusion-models/)<br/>
[Robin Rombach](https://github.com/rromb)\*,
[Andreas Blattmann](https://github.com/ablattmann)\*,
[Dominik Lorenz](https://github.com/qp-qp)\,
[Patrick Esser](https://github.com/pesser),
[Björn Ommer](https://hci.iwr.uni-heidelberg.de/Staff/bommer)<br/>


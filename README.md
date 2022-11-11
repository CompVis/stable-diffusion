# Waifu Diffusion

[Waifu Diffusion](https://huggingface.co/hakurei/waifu-diffusion) is the name for this project of finetuning [Stable Diffusion](https://huggingface.co/runwayml/stable-diffusion-v1-5) on anime-styled images.

<img src=https://user-images.githubusercontent.com/26317155/194690196-8da73f2a-039d-4349-8b08-e24e8fd20959.png width=40% height=40%>

<sub>1girl, aqua eyes, baseball cap, blonde hair, closed mouth, earrings, green background, hat, hoop earrings, jewelry, looking at viewer, shirt, short hair, simple background, solo, upper body, yellow shirt</sub>

## Setup

```shell
pip install -r requirements.txt
```

## Project Structure

```
├── dataset: Dataset preparation and utilities
│   ├── aesthetic: Aesthetic ranking
│   └── download: Downloading utilities
└── trainer: The actual training code
```

## License
Training Code: [AGPL-3.0](LICENSE)
Model Weights: [CreativeML Open RAIL-M](https://huggingface.co/spaces/CompVis/stable-diffusion-license)

[![Discord Server](https://discordapp.com/api/guilds/930499730843250783/widget.png?style=banner2)](https://discord.gg/Sx6Spmsgx7)

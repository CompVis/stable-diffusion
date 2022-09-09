

# Waifu Diffusion

Waifu Diffusion is the name for this project of finetuning Stable Diffusion on Danbooru images.

<img src=https://cdn.discordapp.com/attachments/872361510133981234/1016022078635388979/unknown.png?3867929 width=40% height=40%>

<sub>Prompt: touhou 1girl komeiji_koishi portrait</sub>

[![Discord Server](https://discordapp.com/api/guilds/930499730843250783/widget.png?style=banner2)](https://discord.gg/Sx6Spmsgx7)

## Documentation

[Index](./docs/en/README.md)

[Weights](./docs/en/weights/README.md)

[Training Guide](./docs/en/training/README.md)

All thanks goes to CompVis and Stability AI for releasing this codebase!

Model Link: https://huggingface.co/hakurei/waifu-diffusion

### Any questions? Come hop on by to our Discord server!

[![Discord Server](https://discordapp.com/api/guilds/930499730843250783/widget.png?style=banner2)](https://discord.gg/Sx6Spmsgx7)

# Stable Diffusion
*Stable Diffusion was made possible thanks to a collaboration with [Stability AI](https://stability.ai/) and [Runway](https://runwayml.com/) and builds upon our previous work:*

## Comments 

- Our codebase for the diffusion models builds heavily on [OpenAI's ADM codebase](https://github.com/openai/guided-diffusion)
and [https://github.com/lucidrains/denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch). 
Thanks for open-sourcing!

- The implementation of the transformer encoder is from [x-transformers](https://github.com/lucidrains/x-transformers) by [lucidrains](https://github.com/lucidrains?tab=repositories). 


## BibTeX

```
@misc{rombach2021highresolution,
      title={High-Resolution Image Synthesis with Latent Diffusion Models}, 
      author={Robin Rombach and Andreas Blattmann and Dominik Lorenz and Patrick Esser and Björn Ommer},
      year={2021},
      eprint={2112.10752},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

```



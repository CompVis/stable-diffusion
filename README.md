<h1 align="center">Optimized Stable Diffusion</h1>
<p align="center">
    <img src="https://img.shields.io/github/last-commit/basujindal/stable-diffusion?logo=Python&logoColor=green&style=for-the-badge"/>
        <img src="https://img.shields.io/github/issues/basujindal/stable-diffusion?logo=GitHub&style=for-the-badge"/>
                <img src="https://img.shields.io/github/stars/basujindal/stable-diffusion?logo=GitHub&style=for-the-badge"/>
</p>

This repo is a modified version of the Stable Diffusion repo, optimized to use less VRAM than the original by sacrificing inference speed.

To achieve this, the stable diffusion model is fragmented into four parts which are sent to the GPU only when needed. After the calculation is done, they are moved back to the CPU. This allows us to run a bigger model while requiring less VRAM.

<h1 align="center">Installation</h1>

All the modified files are in the [optimizedSD](optimizedSD) folder, so if you have already cloned the original repository you can just download and copy this folder into the original instead of cloning the entire repo. You can also clone this repo and follow the same installation steps as the original (mainly creating the conda environment and placing the weights at the specified location).

<h1 align="center">Usage</h1>

## img2img

- `img2img` can generate _512x512 images from a prior image and prompt on a 4GB VRAM GPU in under 20 seconds per image_ on an RTX 2060.

- The maximum size that can fit on 6GB GPU (RTX 2060) is around 576x768.

- For example, the following command will generate 20 512x512 images:

`python optimizedSD/optimized_img2img.py --prompt "Austrian alps" --init-img ~/sketch-mountains-input.jpg --strength 0.8 --n_iter 2 --n_samples 10 --H 512 --W 512`

## txt2img

- `txt2img` can generate _512x512 images from a prompt on a 4GB VRAM GPU in under 25 seconds per image_ on an RTX 2060.

- For example, the following command will generate 20 512x512 images:

`python optimizedSD/optimized_txt2img.py --prompt "Cyberpunk style image of a Telsa car reflection in rain" --H 512 --W 512 --seed 27 --n_iter 2 --n_samples 10 --ddim_steps 50`

<h1 align="center">Using the Gradio GUI</h1>

- You can also use the built-in gradio interface for `img2img` & `txt2img` instead of the command line interface. Activate the conda environment and install the latest version of gradio using `pip install gradio`,

- Run img2img using `python3 optimizedSD/img2img_gradio.py` and txt2img using `python3 optimizedSD/txt2img_gradio.py`.

- img2img_gradio.py has a feature to crop input images. Look for the pen symbol in the image box after selecting the image.

<h1 align="center">Arguments</h1>

## `--seed`

**Seed for image generation**, can be used to reproduce previously generated images. Defaults to a random seed if unspecified.

- The code will give the seed number along with each generated image. To generate the same image again, just specify the seed using `--seed` argument. Images are saved with its seed number as its name by default.

- For example if the seed number for an image is `1234` and it's the 55th image in the folder, the image name will be named `seed_1234_00055.png`.

## `--n_samples`

**Batch size/amount of images to generate at once.**

- To get the lowest inference time per image, use the maximum batch size `--n_samples` that can fit on the GPU. Inference time per image will reduce on increasing the batch size, but the required VRAM will increase.

- If you get a CUDA out of memory error, try reducing the batch size `--n_samples`. If it doesn't work, the other option is to reduce the image width `--W` or height `--H` or both.

## `--n_iter`

**Run _x_ amount of times**

- Equivalent to running the script n_iter number of times. Only difference is that the model is loaded only once per n_iter iterations. Unlike `n_samples`, reducing it doesn't have an effect on VRAM required or inference time.

## `--H` & `--W`

**Height & width of the generated image.**

- Both height and width should be a multiple of 64.

## `--turbo`

**Increases inference speed at the cost of extra VRAM usage.**

- Using this argument increases the inference speed by using around 1GB of extra GPU VRAM. It is especially effective when generating a small batch of images (~ 1 to 4) images. It takes under 25 seconds for txt2img and 15 seconds for img2img (on an RTX 2060, excluding the time to load the model). Use it on larger batch sizes if GPU VRAM available.

## `--precision autocast` or `--precision full`

**Whether to use `full` or `mixed` precision**

- Mixed Precision is enabled by default. If you don't have a GPU with tensor cores (any GTX 10 series card), you may not be able use mixed precision. Use the `--precision full` argument to disable it.

## `--format png` or `--format jpg`

**Output image format**

- The default output format is `png`. While `png` is lossless, it takes up a lot of space (unless large portions of the image happen to be a single colour). Use lossy `jpg` to get smaller image file sizes.

## `--unet_bs`

**Batch size for the unet model**

- Takes up a lot of extra RAM for **very little improvement** in inference time. `unet_bs` > 1 is not recommended!

- Should generally be a multiple of 2x(n_samples)

<h1 align="center">Weighted Prompts</h1>

- Prompts can also be weighted to put relative emphasis on certain words.
  eg. `--prompt tabby cat:0.25 white duck:0.75 hybrid`.

- The number followed by the colon represents the weight given to the words before the colon. The weights can be both fractions or integers.

## Changelog

- v0.7: Added support for logging, jpg file format
- v0.6: Added support for using weighted prompts. (based on @lstein's [repo](https://github.com/lstein/stable-diffusion))
- v0.5: Added support for using gradio interface.
- v0.4: Added support for specifying image seed.
- v0.3: Added support for using mixed precision.
- v0.2: Added support for generating images in batches.
- v0.1: Split the model into multiple parts to run it on lower VRAM.

# Update v0.6: Added support for weighted prompts (based on the code from @lstein's [repo](https://github.com/lstein/stable-diffusion))

- You can now use weighted prompts to put relative emphasis on certain words.
  eg. `--prompt tabby cat:0.25 white duck:0.75 hybrid`.
- The number followed by the colon represents the weight given to the words before the colon.
  The weights can be fractions or integers.

# Optimized Stable Diffusion (Sort of)

This repo is a modified version of the Stable Diffusion repo, optimized to use lesser VRAM than the original by sacrificing on inference speed.

img2img to generate new image based on a prior image and prompt

- `optimized_img2img.py` Generate images using CLI
- `img2img_gradio.py` Generate images using gradio GUI

txt2img to generate an image based only on a prompt

- `optimized_txt2img.py` Generate images using CLI
- `txt2img_gradio.py` Generate images using gradio GUI

### img2img

- It can generate _512x512 images from a prior image and prompt on a 4GB VRAM GPU in under 20 seconds per image_ (RTX 2060 in my case).

- You can use the `--H` & `--W` arguments to set the size of the generated image.The maximum size that can fit on 6GB GPU (RTX 2060) is around 576x768.

- For example, the following command will generate 20 512x512 images:

`python optimizedSD/optimized_img2img.py --prompt "Austrian alps" --init-img ~/sketch-mountains-input.jpg --strength 0.8 --n_iter 2 --n_samples 5 --H 576 --W 768`

### txt2img

- It can generate _512x512 images from a prompt on a 4GB VRAM GPU in under 25 seconds per image_ (RTX 2060 in my case).

- You can use the `--H` & `--W` arguments to set the size of the generated image.

- For example, the following command will generate 20 512x512 images:

`python optimizedSD/optimized_txt2img.py --prompt "Cyberpunk style image of a Telsa car reflection in rain" --H 512 --W 512 --seed 27 --n_iter 2 --n_samples 10 --ddim_steps 50`

---

### `--seed` (Seed)

- The code will give the seed number along with each generated image. To generate the same image again, just specify the seed using `--seed` argument. Also, images will be saved with its seed number as its name.

- eg. If the seed number for an image is `1234` and it's the 55th image in the folder, the image name will be named `seed_1234_00055.png`. If no seed is given as an argument, a random initial seed will be choosen.

### `--n_samples` (batch size)

- To get the lowest inference time per image, use the maximum batch size `--n_samples` that can fit on the GPU. Inference time per image will reduce on increasing the batch size, but the required VRAM will also increase.

- If you get a CUDA out of memory error, try reducing the batch size `--n_samples`. If it doesn't work, the other option is to reduce the image width `--W` or height `--H` or both.

### `--H` & `--W` (Height & Width)

- Both height and width should be a multiple of 64

### `--precision autocast` or `--precision full` (Full or Mixed Precision)

- Mixed Precision is enabled by default. If you don't have a GPU with tensor cores, you can still use mixed precision to run the code using lesser VRAM but the inference time may be larger. And if you feel that the inference is slower, try using the `--precision full` argument to disable it.

### Gradio for Graphical User Interface

- You can also use gradio interface for img2img & txt2img instead of the CLI. Just activate the conda env and install the latest version of gradio using `pip install gradio` .

- Run img2img using `python optimizedSD/img2img_gradio` and txt2img using `python optimizedSD/img2img_gradio`.

### Weighted Prompts

- The prompts can also be weighted to put relative emphasis on certain words.
  eg. `--prompt tabby cat:0.25 white duck:0.75 hybrid`.

- The number followed by the colon represents the weight given to the words before the colon.The weights can be both fractions or integers.

### Installation

- All the modified files are in the [optimizedSD](optimizedSD) folder, so if you have already cloned the original repo, you can just download and copy this folder into the orignal repo instead of cloning the entire repo. You can also clone this repo and follow the same installation steps as the original repo(mainly creating the conda env and placing the weights at the specified location).

---

- To achieve this, the stable diffusion model is fragmented into four parts which are sent to the GPU only when needed. After the calculation is done, they are moved back to the CPU. This allows us to run a bigger model on a lower VRAM.

- The only drawback is higher inference time which is still an order of magnitude faster than inference on CPU.

## Changelog

- v0.6: Added support for using weighted prompts. (based on @lstein [repo](https://github.com/lstein/stable-diffusion))
- v0.5: Added support for using gradio interface.
- v0.4: Added support for specifying image seed.
- v0.3: Added support for using mixed precision.
- v0.2: Added support for generating images in batches.
- v0.1: Split the model into multiple parts to run it on lower VRAM.

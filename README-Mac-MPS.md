# Apple Silicon Mac Users

Several people have gotten Stable Diffusion to work on Apple Silicon
Macs using Anaconda, miniforge, etc. I've gathered up most of their instructions and
put them in this fork (and readme). Things have moved really fast and so these
instructions change often. Hopefully things will settle down a little.

There's several places where people are discussing Apple
MPS functionality: [the original CompVis
issue](https://github.com/CompVis/stable-diffusion/issues/25), and generally on
[lstein's fork](https://github.com/lstein/stable-diffusion/).

You have to have macOS 12.3 Monterey or later. Anything earlier than that won't work.

Tested on a 2022 Macbook M2 Air with 10-core GPU and 24 GB unified memory.

How to:

```
git clone https://github.com/lstein/stable-diffusion.git
cd stable-diffusion

mkdir -p models/ldm/stable-diffusion-v1/
PATH_TO_CKPT="$HOME/Documents/stable-diffusion-v-1-4-original"  # or wherever yours is.
ln -s "$PATH_TO_CKPT/sd-v1-4.ckpt" models/ldm/stable-diffusion-v1/model.ckpt

CONDA_SUBDIR=osx-arm64 conda env create -f environment-mac.yaml
conda activate ldm

python scripts/preload_models.py
python scripts/dream.py --full_precision  # half-precision requires autocast and won't work
```

After you follow all the instructions and run dream.py you might get several errors. Here's the errors I've seen and found solutions for.

### Is it slow?

Be sure to specify 1 sample and 1 iteration.

	python ./scripts/orig_scripts/txt2img.py --prompt "ocean" --ddim_steps 5 --n_samples 1 --n_iter 1

### Doesn't work anymore?

PyTorch nightly includes support for MPS. Because of this, this setup is
inherently unstable. One morning I woke up and it no longer worked no matter
what I did until I switched to miniforge. However, I have another Mac that works
just fine with Anaconda. If you can't get it to work, please search a little
first because many of the errors will get posted and solved. If you can't find
a solution please [create an issue](https://github.com/lstein/stable-diffusion/issues).

One debugging step is to update to the latest version of PyTorch nightly.

	conda install pytorch torchvision torchaudio -c pytorch-nightly

Or you can clean everything up.

	conda clean --yes --all

Or you can reset Anaconda.

	conda update --force-reinstall -y -n base -c defaults conda

### "No module named cv2" (or some other module)

Did you remember to `conda activate ldm`? If your terminal prompt
begins with "(ldm)" then you activated it. If it begins with "(base)"
or something else you haven't.

If it says you're missing taming you need to rebuild your virtual
environment.

	conda env remove -n ldm
	conda env create -f environment-mac.yaml

If you have activated the ldm virtual environment and tried rebuilding
it, maybe the problem could be that I have something installed that
you don't and you'll just need to manually install it. Make sure you
activate the virtual environment so it installs there instead of
globally.

	conda activate ldm
	pip install *name*

You might also need to install Rust (I mention this again below).


### Debugging?

Tired of waiting for your renders to finish before you can see if it
works? Reduce the steps! The image quality will be horrible but at least you'll
get quick feedback.

	python ./scripts/txt2img.py --prompt "ocean" --ddim_steps 5 --n_samples 1 --n_iter 1

### OSError: Can't load tokenizer for 'openai/clip-vit-large-patch14'...

	python scripts/preload_models.py

### "The operator [name] is not current implemented for the MPS device." (sic)

Example error.

```
...
NotImplementedError: The operator 'aten::_index_put_impl_' is not current implemented for the MPS device. If you want this op to be added in priority during the prototype phase of this feature, please comment on [https://github.com/pytorch/pytorch/issues/77764](https://github.com/pytorch/pytorch/issues/77764). As a temporary fix, you can set the environment variable `PYTORCH_ENABLE_MPS_FALLBACK=1` to use the CPU as a fallback for this op. WARNING: this will be slower than running natively on MPS.
```

The lstein branch includes this fix in [environment-mac.yaml](https://github.com/lstein/stable-diffusion/blob/main/environment-mac.yaml).

### "Could not build wheels for tokenizers"

I have not seen this error because I had Rust installed on my computer before I started playing with Stable Diffusion. The fix is to install Rust.

	curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

### How come `--seed` doesn't work?

First this:

> Completely reproducible results are not guaranteed across PyTorch
releases, individual commits, or different platforms. Furthermore,
results may not be reproducible between CPU and GPU executions, even
when using identical seeds.

[PyTorch docs](https://pytorch.org/docs/stable/notes/randomness.html)

Second, we might have a fix that at least gets a consistent seed sort of. We're
still working on it.

### libiomp5.dylib error?

	OMP: Error #15: Initializing libiomp5.dylib, but found libomp.dylib already initialized.

You are likely using an Intel package by mistake. Be sure to run conda with
the environment variable `CONDA_SUBDIR=osx-arm64`, like so:

`CONDA_SUBDIR=osx-arm64 conda install ...`

This error happens with Anaconda on Macs when the Intel-only `mkl` is pulled in by
a dependency. [nomkl](https://stackoverflow.com/questions/66224879/what-is-the-nomkl-python-package-used-for)
is a metapackage designed to prevent this, by making it impossible to install
`mkl`, but if your environment is already broken it may not work. 

Do *not* use `os.environ['KMP_DUPLICATE_LIB_OK']='True'` or equivalents as this
masks the underlying issue of using Intel packages.

### Not enough memory.

This seems to be a common problem and is probably the underlying
problem for a lot of symptoms (listed below). The fix is to lower your
image size or to add `model.half()` right after the model is loaded. I
should probably test it out. I've read that the reason this fixes
problems is because it converts the model from 32-bit to 16-bit and
that leaves more RAM for other things. I have no idea how that would
affect the quality of the images though.

See [this issue](https://github.com/CompVis/stable-diffusion/issues/71).

### "Error: product of dimension sizes > 2**31'"

This error happens with img2img, which I haven't played with too much
yet. But I know it's because your image is too big or the resolution
isn't a multiple of 32x32. Because the stable-diffusion model was
trained on images that were 512 x 512, it's always best to use that
output size (which is the default). However, if you're using that size
and you get the above error, try 256 x 256 or 512 x 256 or something
as the source image.

BTW, 2**31-1 = [2,147,483,647](https://en.wikipedia.org/wiki/2,147,483,647#In_computing), which is also 32-bit signed [LONG_MAX](https://en.wikipedia.org/wiki/C_data_types) in C.

### I just got Rickrolled! Do I have a virus?

You don't have a virus. It's part of the project. Here's
[Rick](https://github.com/lstein/stable-diffusion/blob/main/assets/rick.jpeg)
and here's [the
code](https://github.com/lstein/stable-diffusion/blob/69ae4b35e0a0f6ee1af8bb9a5d0016ccb27e36dc/scripts/txt2img.py#L79)
that swaps him in. It's a NSFW filter, which IMO, doesn't work very
good (and we call this "computer vision", sheesh).

Actually, this could be happening because there's not enough RAM. You could try the `model.half()` suggestion or specify smaller output images.

### My images come out black

We might have this fixed, we are still testing.

There's a [similar issue](https://github.com/CompVis/stable-diffusion/issues/69)
on CUDA GPU's where the images come out green. Maybe it's the same issue?
Someone in that issue says to use "--precision full", but this fork
actually disables that flag. I don't know why, someone else provided
that code and I don't know what it does. Maybe the `model.half()`
suggestion above would fix this issue too. I should probably test it.

### "view size is not compatible with input tensor's size and stride"

```
  File "/opt/anaconda3/envs/ldm/lib/python3.10/site-packages/torch/nn/functional.py", line 2511, in layer_norm
    return torch.layer_norm(input, normalized_shape, weight, bias, eps, torch.backends.cudnn.enabled)
RuntimeError: view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.
```

Update to the latest version of lstein/stable-diffusion. We were
patching pytorch but we found a file in stable-diffusion that we could
change instead. This is a 32-bit vs 16-bit problem.

### The processor must support the Intel bla bla bla

What? Intel? On an Apple Silicon?

	Intel MKL FATAL ERROR: This system does not meet the minimum requirements for use of the Intel(R) Math Kernel Library.
	The processor must support the Intel(R) Supplemental Streaming SIMD Extensions 3 (Intel(R) SSSE3) instructions.██████████████| 50/50 [02:25<00:00,  2.53s/it]
	The processor must support the Intel(R) Streaming SIMD Extensions 4.2 (Intel(R) SSE4.2) instructions.
	The processor must support the Intel(R) Advanced Vector Extensions (Intel(R) AVX) instructions.

This is due to the Intel `mkl` package getting picked up when you try to install
something that depends on it-- Rosetta can translate some Intel instructions but
not the specialized ones here. To avoid this, make sure to use the environment
variable `CONDA_SUBDIR=osx-arm64`, which restricts the Conda environment to only
use ARM packages, and use `nomkl` as described above.

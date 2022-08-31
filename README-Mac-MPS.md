# Apple Silicon Mac Users

Several people have gotten Stable Diffusion to work on Apple Silicon
Macs using Anaconda. I've gathered up most of their instructions and
put them in this fork (and readme). I haven't tested anything besides
Anaconda, and I've read about issues with things like miniforge, so if
you have an issue that isn't dealt with in this fork then head on over
to the [Apple
Silicon](https://github.com/CompVis/stable-diffusion/issues/25) issue
on GitHub (that page is so long that GitHub hides most of it by
default, so you need to find the hidden part and expand it to view the
whole thing). This fork would not have been possible without the work
done by the people on that issue.

You have to have macOS 12.3 Monterey or later. Anything earlier than that won't work.

BTW, I haven't tested any of this on Intel Macs.

How to:

```
git clone https://github.com/lstein/stable-diffusion.git
cd stable-diffusion

mkdir -p models/ldm/stable-diffusion-v1/
ln -s /path/to/ckpt/sd-v1-1.ckpt models/ldm/stable-diffusion-v1/model.ckpt

conda env create -f environment-mac.yaml
conda activate ldm
```

These instructions are identical to the main repo except I added
environment-mac.yaml because Mac doesn't have cudatoolkit.

After you follow all the instructions and run txt2img.py you might get several errors. Here's the errors I've seen and found solutions for.

### Doesn't work anymore?

We are using PyTorch nightly, which includes support for MPS. I don't
know exactly how Anaconda does updates, but I woke up one morning and
Stable Diffusion crashed and I couldn't think of anything I did that
would've changed anything the night before, when it worked. A day and
a half later I finally got it working again. I don't know what changed
overnight. PyTorch-nightly changes overnight but I'm pretty sure I
didn't manually update it. Either way, things are probably going to be
bumpy on Apple Silicon until PyTorch releases a firm version that we
can lock to.

To manually update to the latest version of PyTorch nightly (which could fix issues), run this command.

	conda install pytorch torchvision torchaudio -c pytorch-nightly

## Debugging?

Tired of waiting for your renders to finish before you can see if it
works? Reduce the steps! The picture wont look like anything but if it
finishes, hey, it works! This could also help you figure out if you've
got a memory problem, because I'm betting 1 step doesn't use much
memory.

	python ./scripts/txt2img.py --prompt "ocean" --ddim_steps 1

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

### "The operator [name] is not current implemented for the MPS device." (sic)

Example error.

```
...
NotImplementedError: The operator 'aten::index.Tensor' is not current implemented for the MPS device. If you want this op to be added in priority during the prototype phase of this feature, please comment on [https://github.com/pytorch/pytorch/issues/77764](https://github.com/pytorch/pytorch/issues/77764). As a temporary fix, you can set the environment variable `PYTORCH_ENABLE_MPS_FALLBACK=1` to use the CPU as a fallback for this op. WARNING: this will be slower than running natively on MPS.
```

Just do what it says:

	export PYTORCH_ENABLE_MPS_FALLBACK=1

### "Could not build wheels for tokenizers"

I have not seen this error because I had Rust installed on my computer before I started playing with Stable Diffusion. The fix is to install Rust.

	curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

### How come `--seed` doesn't work?

> Completely reproducible results are not guaranteed across PyTorch
releases, individual commits, or different platforms. Furthermore,
results may not be reproducible between CPU and GPU executions, even
when using identical seeds.

[PyTorch docs](https://pytorch.org/docs/stable/notes/randomness.html)

There is an [open issue](https://github.com/pytorch/pytorch/issues/78035) (as of August 2022) in pytorch regarding gradient inconsistency. I am guessing that's what is causing this.

### libiomp5.dylib error?

	OMP: Error #15: Initializing libiomp5.dylib, but found libomp.dylib already initialized.

There are several things you can do. First, you could use something
besides Anaconda like miniforge. I read a lot of things online telling
people to use something else, but I am stuck with Anaconda for other
reasons.

Or you can try this.

	export KMP_DUPLICATE_LIB_OK=True

Or this (which takes forever on my computer and didn't work anyway).

	conda install nomkl

This error happens with Anaconda on Macs, and
[nomkl](https://stackoverflow.com/questions/66224879/what-is-the-nomkl-python-package-used-for)
is supposed to fix the issue (it isn't a module but a fix of some
sort). [There's more
suggestions](https://stackoverflow.com/questions/53014306/error-15-initializing-libiomp5-dylib-but-found-libiomp5-dylib-already-initial),
like uninstalling tensorflow and reinstalling. I haven't tried them.

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

I haven't solved this issue. I just throw away my black
images. There's a [similar
issue](https://github.com/CompVis/stable-diffusion/issues/69) on CUDA
GPU's where the images come out green. Maybe it's the same issue?
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

This fixed it for me:

	conda clean --yes --all

### Still slow?

I changed the defaults of n_samples and n_iter to 1 so that it uses
less RAM and makes less images so it will be faster the first time you
use it. I don't actually know what n_samples does internally, but I
know it consumes a lot more RAM. The n_iter flag just loops around the
image creation code, so it shouldn't consume more RAM (it should be
faster if you're going to do multiple images because the libraries and
model will already be loaded--use a prompt file to get this speed
boost).

These flags are the default sample and iter settings in this fork/branch:

~~~~
python scripts/txt2img.py --prompt "ocean" --n_samples=1 --n_iter=1
~~~



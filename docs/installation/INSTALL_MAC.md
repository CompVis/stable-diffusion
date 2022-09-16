---
title: macOS
---

## Requirements

- macOS 12.3 Monterey or later
- Python
- Patience
- Apple Silicon or Intel Mac

Things have moved really fast and so these instructions change often and are
often out-of-date. One of the problems is that there are so many different ways
to run this.

We are trying to build a testing setup so that when we make changes it doesn't
always break.

How to (this hasn't been 100% tested yet):

First get the weights checkpoint download started - it's big:

1. Sign up at https://huggingface.co
2. Go to the
   [Stable diffusion diffusion model page](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original)
3. Accept the terms and click Access Repository:
4. Download
   [sd-v1-4.ckpt (4.27 GB)](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/blob/main/sd-v1-4.ckpt)
   and note where you have saved it (probably the Downloads folder)

   While that is downloading, open Terminal and run the following commands one
   at a time.

```bash
# install brew (and Xcode command line tools):

/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Now there are two different routes to get the Python (miniconda) environment up and running:
# 1. Alongside pyenv
# 2. No pyenv
#
# If you don't know what we are talking about, choose 2.
#
# NOW EITHER DO
# 1. Installing alongside pyenv

  brew install pyenv-virtualenv # you might have this from before, no problem
  pyenv install anaconda3-2022.05
  pyenv virtualenv anaconda3-2022.05
  eval "$(pyenv init -)"
  pyenv activate anaconda3-2022.05

# OR,
# 2. Installing standalone
# install python 3, git, cmake, protobuf:
brew install cmake protobuf rust

# install miniconda for M1 arm64:
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh -o Miniconda3-latest-MacOSX-arm64.sh
/bin/bash Miniconda3-latest-MacOSX-arm64.sh

# OR install miniconda for Intel:
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -o Miniconda3-latest-MacOSX-x86_64.sh
/bin/bash Miniconda3-latest-MacOSX-x86_64.sh


# EITHER WAY,
# continue from here

# clone the repo
  git clone https://github.com/lstein/stable-diffusion.git
  cd stable-diffusion

#
# wait until the checkpoint file has downloaded, then proceed
#

# create symlink to checkpoint
  mkdir -p models/ldm/stable-diffusion-v1/

  PATH_TO_CKPT="$HOME/Downloads"  # or wherever you saved sd-v1-4.ckpt

  ln -s "$PATH_TO_CKPT/sd-v1-4.ckpt" models/ldm/stable-diffusion-v1/model.ckpt

# install packages for arm64
PIP_EXISTS_ACTION=w CONDA_SUBDIR=osx-arm64 conda env create -f environment-mac.yaml
conda activate ldm

# OR install packages for x86_64
PIP_EXISTS_ACTION=w CONDA_SUBDIR=osx-x86_64 conda env create -f environment-mac.yaml
conda activate ldm

# only need to do this once
python scripts/preload_models.py

# run SD!
python scripts/dream.py --full_precision  # half-precision requires autocast and won't work

# or run the web interface!
python scripts/dream.py --web
```

The original scripts should work as well.

```bash
python scripts/orig_scripts/txt2img.py --prompt "a photograph of an astronaut riding a horse" --plms
```

Note,

```bash
export PIP_EXISTS_ACTION=w
```

is a precaution to fix

```bash
conda env create -f environment-mac.yaml
```

never finishing in some situations. So it isn't required but wont hurt.

After you follow all the instructions and run dream.py you might get several
errors. Here's the errors I've seen and found solutions for.

---

### Is it slow?

Be sure to specify 1 sample and 1 iteration.

```bash
python ./scripts/orig_scripts/txt2img.py \
      --prompt "ocean" \
      --ddim_steps 5 \
      --n_samples 1 \
      --n_iter 1
```

---

### Doesn't work anymore?

PyTorch nightly includes support for MPS. Because of this, this setup is
inherently unstable. One morning I woke up and it no longer worked no matter
what I did until I switched to miniforge. However, I have another Mac that works
just fine with Anaconda. If you can't get it to work, please search a little
first because many of the errors will get posted and solved. If you can't find a
solution please
[create an issue](https://github.com/lstein/stable-diffusion/issues).

One debugging step is to update to the latest version of PyTorch nightly.

```bash
conda install pytorch torchvision torchaudio -c pytorch-nightly
```

If it takes forever to run

```bash
conda env create -f environment-mac.yaml
```

you could try to run `git clean -f` followed by:

`conda clean --yes --all`

Or you could try to completley reset Anaconda:

```bash
conda update --force-reinstall -y -n base -c defaults conda
```

---

### "No module named cv2", torch, 'ldm', 'transformers', 'taming', etc

There are several causes of these errors.

- First, did you remember to `conda activate ldm`? If your terminal prompt
  begins with "(ldm)" then you activated it. If it begins with "(base)" or
  something else you haven't.

- Second, you might've run `./scripts/preload_models.py` or `./scripts/dream.py`
  instead of `python ./scripts/preload_models.py` or
  `python ./scripts/dream.py`. The cause of this error is long so it's below.

- Third, if it says you're missing taming you need to rebuild your virtual
  environment.

````bash
conda deactivate

conda env remove -n ldm
PIP_EXISTS_ACTION=w CONDA_SUBDIR=osx-arm64 conda env create -f environment-mac.yaml
```

Fourth, If you have activated the ldm virtual environment and tried rebuilding
it, maybe the problem could be that I have something installed that you don't
and you'll just need to manually install it. Make sure you activate the virtual
environment so it installs there instead of globally.

`conda activate ldm pip install _name_`

You might also need to install Rust (I mention this again below).

---

### How many snakes are living in your computer?

You might have multiple Python installations on your system, in which case it's
important to be explicit and consistent about which one to use for a given
project. This is because virtual environments are coupled to the Python that
created it (and all the associated 'system-level' modules).

When you run `python` or `python3`, your shell searches the colon-delimited
locations in the `PATH` environment variable (`echo $PATH` to see that list) in
that order - first match wins. You can ask for the location of the first
`python3` found in your `PATH` with the `which` command like this:

```bash
% which python3
/usr/bin/python3
```

Anything in `/usr/bin` is
[part of the OS](https://developer.apple.com/library/archive/documentation/FileManagement/Conceptual/FileSystemProgrammingGuide/FileSystemOverview/FileSystemOverview.html#//apple_ref/doc/uid/TP40010672-CH2-SW6).
However, `/usr/bin/python3` is not actually python3, but rather a stub that
offers to install Xcode (which includes python 3). If you have Xcode installed
already, `/usr/bin/python3` will execute
`/Library/Developer/CommandLineTools/usr/bin/python3` or
`/Applications/Xcode.app/Contents/Developer/usr/bin/python3` (depending on which
Xcode you've selected with `xcode-select`).

Note that `/usr/bin/python` is an entirely different python - specifically,
python 2. Note: starting in macOS 12.3, `/usr/bin/python` no longer exists.

```bash
% which python3
/opt/homebrew/bin/python3
```

If you installed python3 with Homebrew and you've modified your path to search
for Homebrew binaries before system ones, you'll see the above path.

```bash
% which python
/opt/anaconda3/bin/python
```

If you have Anaconda installed, you will see the above path. There is a
`/opt/anaconda3/bin/python3` also.

We expect that `/opt/anaconda3/bin/python` and `/opt/anaconda3/bin/python3`
should actually be the _same python_, which you can verify by comparing the
output of `python3 -V` and `python -V`.

```bash
(ldm) % which python
/Users/name/miniforge3/envs/ldm/bin/python
```

The above is what you'll see if you have miniforge and you've correctly
activated the ldm environment, and you used option 2 in the setup instructions
above ("no pyenv").

```bash
(anaconda3-2022.05) % which python
/Users/name/.pyenv/shims/python
```

... and the above is what you'll see if you used option 1 ("Alongside pyenv").

It's all a mess and you should know
[how to modify the path environment variable](https://support.apple.com/guide/terminal/use-environment-variables-apd382cc5fa-4f58-4449-b20a-41c53c006f8f/mac)
if you want to fix it. Here's a brief hint of all the ways you can modify it
(don't really have the time to explain it all here).

- ~/.zshrc
- ~/.bash_profile
- ~/.bashrc
- /etc/paths.d
- /etc/path

Which one you use will depend on what you have installed except putting a file
in /etc/paths.d is what I prefer to do.

Finally, to answer the question posed by this section's title, it may help to
list all of the `python` / `python3` things found in `$PATH` instead of just the
one that will be executed by default. To do that, add the `-a` switch to
`which`:

    % which -a python3
    ...

### Debugging?

Tired of waiting for your renders to finish before you can see if it works?
Reduce the steps! The image quality will be horrible but at least you'll get
quick feedback.

    python ./scripts/txt2img.py --prompt "ocean" --ddim_steps 5 --n_samples 1 --n_iter 1

### OSError: Can't load tokenizer for 'openai/clip-vit-large-patch14'...

    python scripts/preload_models.py

### "The operator [name] is not current implemented for the MPS device." (sic)

Example error.

```

... NotImplementedError: The operator 'aten::_index_put_impl_' is not current
implemented for the MPS device. If you want this op to be added in priority
during the prototype phase of this feature, please comment on
[https://github.com/pytorch/pytorch/issues/77764](https://github.com/pytorch/pytorch/issues/77764).
As a temporary fix, you can set the environment variable
`PYTORCH_ENABLE_MPS_FALLBACK=1` to use the CPU as a fallback for this op.
WARNING: this will be slower than running natively on MPS.

```

The lstein branch includes this fix in
[environment-mac.yaml](https://github.com/lstein/stable-diffusion/blob/main/environment-mac.yaml).

### "Could not build wheels for tokenizers"

I have not seen this error because I had Rust installed on my computer before I
started playing with Stable Diffusion. The fix is to install Rust.

    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

### How come `--seed` doesn't work?

First this:

> Completely reproducible results are not guaranteed across PyTorch releases,
> individual commits, or different platforms. Furthermore, results may not be
> reproducible between CPU and GPU executions, even when using identical seeds.

[PyTorch docs](https://pytorch.org/docs/stable/notes/randomness.html)

Second, we might have a fix that at least gets a consistent seed sort of. We're
still working on it.

### libiomp5.dylib error?

    OMP: Error #15: Initializing libiomp5.dylib, but found libomp.dylib already initialized.

You are likely using an Intel package by mistake. Be sure to run conda with the
environment variable `CONDA_SUBDIR=osx-arm64`, like so:

`CONDA_SUBDIR=osx-arm64 conda install ...`

This error happens with Anaconda on Macs when the Intel-only `mkl` is pulled in
by a dependency.
[nomkl](https://stackoverflow.com/questions/66224879/what-is-the-nomkl-python-package-used-for)
is a metapackage designed to prevent this, by making it impossible to install
`mkl`, but if your environment is already broken it may not work.

Do _not_ use `os.environ['KMP_DUPLICATE_LIB_OK']='True'` or equivalents as this
masks the underlying issue of using Intel packages.

### Not enough memory

This seems to be a common problem and is probably the underlying problem for a
lot of symptoms (listed below). The fix is to lower your image size or to add
`model.half()` right after the model is loaded. I should probably test it out.
I've read that the reason this fixes problems is because it converts the model
from 32-bit to 16-bit and that leaves more RAM for other things. I have no idea
how that would affect the quality of the images though.

See [this issue](https://github.com/CompVis/stable-diffusion/issues/71).

### "Error: product of dimension sizes > 2\*\*31'"

This error happens with img2img, which I haven't played with too much yet. But I
know it's because your image is too big or the resolution isn't a multiple of
32x32. Because the stable-diffusion model was trained on images that were 512 x
512, it's always best to use that output size (which is the default). However,
if you're using that size and you get the above error, try 256 x 256 or 512 x
256 or something as the source image.

BTW, 2\*\*31-1 =
[2,147,483,647](https://en.wikipedia.org/wiki/2,147,483,647#In_computing), which
is also 32-bit signed [LONG_MAX](https://en.wikipedia.org/wiki/C_data_types) in
C.

### I just got Rickrolled! Do I have a virus?

You don't have a virus. It's part of the project. Here's
[Rick](https://github.com/lstein/stable-diffusion/blob/main/assets/rick.jpeg)
and here's
[the code](https://github.com/lstein/stable-diffusion/blob/69ae4b35e0a0f6ee1af8bb9a5d0016ccb27e36dc/scripts/txt2img.py#L79)
that swaps him in. It's a NSFW filter, which IMO, doesn't work very good (and we
call this "computer vision", sheesh).

Actually, this could be happening because there's not enough RAM. You could try
the `model.half()` suggestion or specify smaller output images.

### My images come out black

We might have this fixed, we are still testing.

There's a [similar issue](https://github.com/CompVis/stable-diffusion/issues/69)
on CUDA GPU's where the images come out green. Maybe it's the same issue?
Someone in that issue says to use "--precision full", but this fork actually
disables that flag. I don't know why, someone else provided that code and I
don't know what it does. Maybe the `model.half()` suggestion above would fix
this issue too. I should probably test it.

### "view size is not compatible with input tensor's size and stride"

```bash
File "/opt/anaconda3/envs/ldm/lib/python3.10/site-packages/torch/nn/functional.py", line 2511, in layer_norm
return torch.layer_norm(input, normalized_shape, weight, bias, eps, torch.backends.cudnn.enabled)
RuntimeError: view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.
```

Update to the latest version of lstein/stable-diffusion. We were patching
pytorch but we found a file in stable-diffusion that we could change instead.
This is a 32-bit vs 16-bit problem.

---

### The processor must support the Intel bla bla bla

What? Intel? On an Apple Silicon?
`bash Intel MKL FATAL ERROR: This system does not meet the minimum requirements for use of the Intel(R) Math Kernel Library. The processor must support the Intel(R) Supplemental Streaming SIMD Extensions 3 (Intel(R) SSSE3) instructions. The processor must support the Intel(R) Streaming SIMD Extensions 4.2 (Intel(R) SSE4.2) instructions. The processor must support the Intel(R) Advanced Vector Extensions (Intel(R) AVX) instructions. `

This is due to the Intel `mkl` package getting picked up when you try to install
something that depends on it-- Rosetta can translate some Intel instructions but
not the specialized ones here. To avoid this, make sure to use the environment
variable `CONDA_SUBDIR=osx-arm64`, which restricts the Conda environment to only
use ARM packages, and use `nomkl` as described above.

---

### input types 'tensor<2x1280xf32>' and 'tensor<\*xf16>' are not broadcast compatible

May appear when just starting to generate, e.g.:

```bash
dream> clouds
Generating:   0%|                                                              | 0/1 [00:00<?, ?it/s]/Users/[...]/dev/stable-diffusion/ldm/modules/embedding_manager.py:152: UserWarning: The operator 'aten::nonzero' is not currently supported on the MPS backend and will fall back to run on the CPU. This may have performance implications. (Triggered internally at /Users/runner/work/_temp/anaconda/conda-bld/pytorch_1662016319283/work/aten/src/ATen/mps/MPSFallback.mm:11.)
  placeholder_idx = torch.where(
                                                                                                    loc("mps_add"("(mpsFileLoc): /AppleInternal/Library/BuildRoots/20d6c351-ee94-11ec-bcaf-7247572f23b4/Library/Caches/com.apple.xbs/Sources/MetalPerformanceShadersGraph/mpsgraph/MetalPerformanceShadersGraph/Core/Files/MPSGraphUtilities.mm":219:0)): error: input types 'tensor<2x1280xf32>' and 'tensor<*xf16>' are not broadcast compatible
LLVM ERROR: Failed to infer result type(s).
Abort trap: 6
/Users/[...]/opt/anaconda3/envs/ldm/lib/python3.9/multiprocessing/resource_tracker.py:216: UserWarning: resource_tracker: There appear to be 1 leaked semaphore objects to clean up at shutdown
  warnings.warn('resource_tracker: There appear to be %d '
```

Macs do not support `autocast/mixed-precision`, so you need to supply
`--full_precision` to use float32 everywhere.

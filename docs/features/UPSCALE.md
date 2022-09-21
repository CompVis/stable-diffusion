---
title: Upscale
---

## Intro

The script provides the ability to restore faces and upscale. You can apply these operations
at the time you generate the images, or at any time to a previously-generated PNG file, using 
the [!fix](#fixing-previously-generated-images) command.

## Face Fixing

The default face restoration module is GFPGAN. The default upscale is Real-ESRGAN. For an alternative
face restoration module, see [CodeFormer Support] below.

As of version 1.14, environment.yaml will install the Real-ESRGAN package into the standard install
location for python packages, and will put GFPGAN into a subdirectory of "src" in the
stable-diffusion directory. (The reason for this is that the standard GFPGAN distribution has a
minor bug that adversely affects image color.) Upscaling with Real-ESRGAN should "just work" without
further intervention. Simply pass the --upscale (-U) option on the dream> command line, or indicate
the desired scale on the popup in the Web GUI.

For **GFPGAN** to work, there is one additional step needed. You will need to download and copy the
GFPGAN [models file](https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth)
into **src/gfpgan/experiments/pretrained_models**. On Mac and Linux systems, here's how you'd do it
using **wget**:

```bash
> wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth src/gfpgan/experiments/pretrained_models/
```

Make sure that you're in the stable-diffusion directory when you do this.

Alternatively, if you have GFPGAN installed elsewhere, or if you are using an earlier version of
this package which asked you to install GFPGAN in a sibling directory, you may use the
`--gfpgan_dir` argument with `dream.py` to set a custom path to your GFPGAN directory. _There are
other GFPGAN related boot arguments if you wish to customize further._

!!! warning "Internet connection needed"

    Users whose GPU machines are isolated from the Internet (e.g.
    on a University cluster) should be aware that the first time you run dream.py with GFPGAN and
    Real-ESRGAN turned on, it will try to download model files from the Internet. To rectify this, you
    may run `python3 scripts/preload_models.py` after you have installed GFPGAN and all its
    dependencies.

## Usage

You will now have access to two new prompt arguments.

### Upscaling

`-U : <upscaling_factor> <upscaling_strength>`

The upscaling prompt argument takes two values. The first value is a scaling factor and should be
set to either `2` or `4` only. This will either scale the image 2x or 4x respectively using
different models.

You can set the scaling stength between `0` and `1.0` to control intensity of the of the scaling.
This is handy because AI upscalers generally tend to smooth out texture details. If you wish to
retain some of those for natural looking results, we recommend using values between `0.5 to 0.8`.

If you do not explicitly specify an upscaling_strength, it will default to 0.75.

### Face Restoration

`-G : <gfpgan_strength>`

This prompt argument controls the strength of the face restoration that is being applied. Similar to
upscaling, values between `0.5 to 0.8` are recommended.

You can use either one or both without any conflicts. In cases where you use both, the image will be
first upscaled and then the face restoration process will be executed to ensure you get the highest
quality facial features.

`--save_orig`

When you use either `-U` or `-G`, the final result you get is upscaled or face modified. If you want
to save the original Stable Diffusion generation, you can use the `-save_orig` prompt argument to
save the original unaffected version too.

### Example Usage

```bash
dream> superman dancing with a panda bear -U 2 0.6 -G 0.4
```

This also works with img2img:

```bash
dream> a man wearing a pineapple hat -I path/to/your/file.png -U 2 0.5 -G 0.6
```

!!! note

    GFPGAN and Real-ESRGAN are both memory intensive. In order to avoid crashes and memory overloads
    during the Stable Diffusion process, these effects are applied after Stable Diffusion has completed
    its work.

    In single image generations, you will see the output right away but when you are using multiple
    iterations, the images will first be generated and then upscaled and face restored after that
    process is complete. While the image generation is taking place, you will still be able to preview
    the base images.

If you wish to stop during the image generation but want to upscale or face restore a particular
generated image, pass it again with the same prompt and generated seed along with the `-U` and `-G`
prompt arguments to perform those actions.

## CodeFormer Support

This repo also allows you to perform face restoration using
[CodeFormer](https://github.com/sczhou/CodeFormer).

In order to setup CodeFormer to work, you need to download the models like with GFPGAN. You can do
this either by running `preload_models.py` or by manually downloading the
[model file](https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth) and
saving it to `ldm/restoration/codeformer/weights` folder.

You can use `-ft` prompt argument to swap between CodeFormer and the default GFPGAN. The above
mentioned `-G` prompt argument will allow you to control the strength of the restoration effect.

### Usage:

The following command will perform face restoration with CodeFormer instead of the default gfpgan.

`<prompt> -G 0.8 -ft codeformer`

### Other Options:

- `-cf` - cf or CodeFormer Fidelity takes values between `0` and `1`. 0 produces high quality
  results but low accuracy and 1 produces lower quality results but higher accuacy to your original
  face.

The following command will perform face restoration with CodeFormer. CodeFormer will output a result
that is closely matching to the input face.

`<prompt> -G 1.0 -ft codeformer -cf 0.9`

The following command will perform face restoration with CodeFormer. CodeFormer will output a result
that is the best restoration possible. This may deviate slightly from the original face. This is an
excellent option to use in situations when there is very little facial data to work with.

`<prompt> -G 1.0 -ft codeformer -cf 0.1`

## Fixing Previously-Generated Images

It is easy to apply face restoration and/or upscaling to any previously-generated file. Just use the
syntax `!fix path/to/file.png <options>`. For example, to apply GFPGAN at strength 0.8 and upscale 2X
for a file named `./outputs/img-samples/000044.2945021133.png`, just run:

~~~~
dream> !fix ./outputs/img-samples/000044.2945021133.png -G 0.8 -U 2
~~~~

A new file named `000044.2945021133.fixed.png` will be created in the output directory. Note that
the `!fix` command does not replace the original file, unlike the behavior at generate time.

### Disabling:

If, for some reason, you do not wish to load the GFPGAN and/or ESRGAN libraries, you can disable them 
on the dream.py command line with the `--no_restore` and `--no_upscale` options, respectively.

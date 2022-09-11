# **Image-to-Image**

This script also provides an img2img feature that lets you seed your
creations with an initial drawing or photo. This is a really cool
feature that tells stable diffusion to build the prompt on top of the
image you provide, preserving the original's basic shape and
layout. To use it, provide the `--init_img` option as shown here:

```
dream> "waterfall and rainbow" --init_img=./init-images/crude_drawing.png --strength=0.5 -s100 -n4
```

The `--init_img (-I)` option gives the path to the seed
picture. `--strength (-f)` controls how much the original will be
modified, ranging from `0.0` (keep the original intact), to `1.0`
(ignore the original completely). The default is `0.75`, and ranges
from `0.25-0.75` give interesting results.

You may also pass a `-v<count>` option to generate count variants on
the original image. This is done by passing the first generated image
back into img2img the requested number of times. It generates
interesting variants.

If the initial image contains transparent regions, then Stable
Diffusion will only draw within the transparent regions, a process
called "inpainting". However, for this to work correctly, the color
information underneath the transparent needs to be preserved, not
erased. See [Creating Transparent Images For
Inpainting](./INPAINTING.md#creating-transparent-regions-for-inpainting)
for details.

# **Embiggen -- upscale your images on limited memory machines**

GFPGAN and Real-ESRGAN are both memory intensive. In order to avoid
crashes and memory overloads during the Stable Diffusion process,
these effects are applied after Stable Diffusion has completed its
work.

In single image generations, you will see the output right away but
when you are using multiple iterations, the images will first be
generated and then upscaled and face restored after that process is
complete. While the image generation is taking place, you will still
be able to preview the base images.

If you wish to stop during the image generation but want to upscale or
face restore a particular generated image, pass it again with the same
prompt and generated seed along with the `-U` and `-G` prompt
arguments to perform those actions.

## Embiggen 

If you wanted to be able to do more (pixels) without running out of VRAM,
or you want to upscale with details that couldn't possibly appear
without the context of a prompt, this is the feature to try out.

Embiggen automates the process of taking an init image, upscaling it,
cutting it into smaller tiles that slightly overlap, running all the
tiles through img2img to refine details with respect to the prompt,
and "stitching" the tiles back together into a cohesive image.

It automatically computes how many tiles are needed, and so it can be fed
*ANY* size init image and perform Img2Img on it (though it will be run only
one tile at a time, which can cause problems, see the Note at the end).

If you're familiar with "GoBig" (ala [progrock-stable](https://github.com/lowfuel/progrock-stable))
it's similar to that, except it can work up to an arbitrarily large size
(instead of just 2x), with tile overlaps configurable as a ratio, and
has extra logic to re-run any number of the tile sub-sections of the image
if for example a small part of a huge run got messed up.

**Usage**

`-embiggen <scaling_factor> <esrgan_strength> <overlap_ratio OR overlap_pixels>`

Takes a scaling factor relative to the size of the `--init_img` (`-I`), followed by
ESRGAN upscaling strength (0 - 1.0), followed by minimum amount of overlap
between tiles as a decimal ratio (0 - 1.0) *OR* a number of pixels.

The scaling factor is how much larger than the `--init_img` the output
should be, and will multiply both x and y axis, so an image that is a
scaling factor of 3.0 has 3*3= 9 times as many pixels, and will take
(at least) 9 times as long (see overlap for why it might be
longer). If the `--init_img` is already the right size `-embiggen 1`,
and it can also be less than one if the init_img is too big.

Esrgan_strength defaults to 0.75, and the overlap_ratio defaults to
0.25, both are optional.


Unlike Img2Img, the `--width` (`-W`) and `--height` (`-H`) arguments
do not control the size of the image as a whole, but the size of the
tiles used to Embiggen the image.

ESRGAN is used to upscale the `--init_img` prior to cutting it into
tiles/pieces to run through img2img and then stitch back
together. Embiggen can be run without ESRGAN; just set the strength to
zero (e.g. `-embiggen 1.75 0`). The output of Embiggen can also be
upscaled after it's finished (`-U`).

The overlap is the minimum that tiles will overlap with adjacent
tiles, specified as either a ratio or a number of pixels. How much the
tiles overlap determines the likelihood the tiling will be noticable,
really small overlaps (e.g. a couple of pixels) may produce noticeable
grid-like fuzzy distortions in the final stitched image. Though, as
the overlapping space doesn't contribute to making the image bigger,
and the larger the overlap the more tiles (and the more time) it will
take to finish.

Because the overlapping parts of tiles don't "contribute" to
increasing size, every tile after the first in a row or column
effectively only covers an extra `1 - overlap_ratio` on each axis. If
the input/`--init_img` is same size as a tile, the ideal (for time)
scaling factors with the default overlap (0.25) are 1.75, 2.5, 3.25,
4.0 etc..

`-embiggen_tiles <spaced list of tiles>`

An advanced usage useful if you only want to alter parts of the image
while running Embiggen. It takes a list of tiles by number to run and
replace onto the initial image e.g. `1 3 5`. It's useful for either
fixing problem spots from a previous Embiggen run, or selectively
altering the prompt for sections of an image - for creative or
coherency reasons.

Tiles are numbered starting with one, and left-to-right,
top-to-bottom.  So, if you are generating a 3x3 tiled image, the
middle row would be `4 5 6`.

**Example Usage**

Running Embiggen with 512x512 tiles on an existing image, scaling up by a factor of 2.5x;
and doing the same again (default ESRGAN strength is 0.75, default overlap between tiles is 0.25):

```
dream > a photo of a forest at sunset -s 100 -W 512 -H 512 -I outputs/forest.png -f 0.4 -embiggen 2.5
dream > a photo of a forest at sunset -s 100 -W 512 -H 512 -I outputs/forest.png -f 0.4 -embiggen 2.5 0.75 0.25
```

If your starting image was also 512x512 this should have taken 9 tiles.

If there weren't enough clouds in the sky of that forest you just made
(and that image is about 1280 pixels (512*2.5) wide A.K.A. three
512x512 tiles with 0.25 overlaps wide) we can replace that top row of
tiles:

```
dream> a photo of puffy clouds over a forest at sunset -s 100 -W 512 -H 512 -I outputs/000002.seed.png -f 0.5 -embiggen_tiles 1 2 3
```

**Note**

Because the same prompt is used on all the tiled images, and the model
doesn't have the context of anything outside the tile being run - it
can end up creating repeated pattern (also called 'motifs') across all
the tiles based on that prompt. The best way to combat this is
lowering the `--strength` (`-f`) to stay more true to the init image,
and increasing the number of steps so there is more compute-time to
create the detail.  Anecdotally `--strength` 0.35-0.45 works pretty
well on most things. It may also work great in some examples even with
the `--strength` set high for patterns, landscapes, or subjects that
are more abstract. Because this is (relatively) fast, you can also
always create a few Embiggen'ed images and manually composite them to
preserve the best parts from each.

Author: [Travco](https://github.com/travco)
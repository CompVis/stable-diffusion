---
title: Changelog
---

# :octicons-log-16: **Changelog**

## v2.0.1 (13 October 2022)

- fix noisy images at high step count when using k* samplers
- dream.py script now calls invoke.py module directly rather than
    via a new python process (which could break the environment)

## v2.0.0 <small>(9 October 2022)</small>

- `dream.py` script renamed `invoke.py`. A `dream.py` script wrapper remains
    for backward compatibility.
- Completely new WebGUI - launch with `python3 scripts/invoke.py --web`
- Support for [inpainting](features/INPAINTING.md) and [outpainting](features/OUTPAINTING.md)
- img2img runs on all k* samplers
- Support for [negative prompts](features/PROMPTS.md#negative-and-unconditioned-prompts)
- Support for CodeFormer face reconstruction
- Support for Textual Inversion on Macintoshes
- Support in both WebGUI and CLI for [post-processing of previously-generated images](features/POSTPROCESS.md)
    using facial reconstruction, ESRGAN upscaling, outcropping (similar to DALL-E infinite canvas),
    and "embiggen" upscaling. See the `!fix` command.
- New `--hires` option on `invoke>` line allows [larger images to be created without duplicating elements](features/CLI.md#this-is-an-example-of-txt2img), at the cost of some performance.
- New `--perlin` and `--threshold` options allow you to add and control variation
    during image generation (see [Thresholding and Perlin Noise Initialization](features/OTHER.md#thresholding-and-perlin-noise-initialization-options))
- Extensive metadata now written into PNG files, allowing reliable regeneration of images
    and tweaking of previous settings.
- Command-line completion in `invoke.py` now works on Windows, Linux and Mac platforms.
- Improved [command-line completion behavior](features/CLI.md)
    New commands added:
  - List command-line history with `!history`
  - Search command-line history with `!search`
  - Clear history with `!clear`
- Deprecated `--full_precision` / `-F`. Simply omit it and `invoke.py` will auto
    configure. To switch away from auto use the new flag like `--precision=float32`.

## v1.14 <small>(11 September 2022)</small>

- Memory optimizations for small-RAM cards. 512x512 now possible on 4 GB GPUs.
- Full support for Apple hardware with M1 or M2 chips.
- Add "seamless mode" for circular tiling of image. Generates beautiful effects.
    ([prixt](https://github.com/prixt)).
- Inpainting support.
- Improved web server GUI.
- Lots of code and documentation cleanups.

## v1.13 <small>(3 September 2022)</small>

- Support image variations (see [VARIATIONS](features/VARIATIONS.md)
    ([Kevin Gibbons](https://github.com/bakkot) and many contributors and reviewers)
- Supports a Google Colab notebook for a standalone server running on Google hardware
    [Arturo Mendivil](https://github.com/artmen1516)
- WebUI supports GFPGAN/ESRGAN facial reconstruction and upscaling
    [Kevin Gibbons](https://github.com/bakkot)
- WebUI supports incremental display of in-progress images during generation
    [Kevin Gibbons](https://github.com/bakkot)
- A new configuration file scheme that allows new models (including upcoming
    stable-diffusion-v1.5) to be added without altering the code.
    ([David Wager](https://github.com/maddavid12))
- Can specify --grid on invoke.py command line as the default.
- Miscellaneous internal bug and stability fixes.
- Works on M1 Apple hardware.
- Multiple bug fixes.

---

## v1.12 <small>(28 August 2022)</small>

- Improved file handling, including ability to read prompts from standard input.
  (kudos to [Yunsaki](https://github.com/yunsaki)
- The web server is now integrated with the invoke.py script. Invoke by adding --web to
  the invoke.py command arguments.
- Face restoration and upscaling via GFPGAN and Real-ESGAN are now automatically
  enabled if the GFPGAN directory is located as a sibling to Stable Diffusion.
  VRAM requirements are modestly reduced. Thanks to both [Blessedcoolant](https://github.com/blessedcoolant) and
  [Oceanswave](https://github.com/oceanswave) for their work on this.
- You can now swap samplers on the invoke> command line. [Blessedcoolant](https://github.com/blessedcoolant)

---

## v1.11 <small>(26 August 2022)</small>

- NEW FEATURE: Support upscaling and face enhancement using the GFPGAN module. (kudos to [Oceanswave](https://github.com/Oceanswave)
- You now can specify a seed of -1 to use the previous image's seed, -2 to use the seed for the image generated before that, etc.
  Seed memory only extends back to the previous command, but will work on all images generated with the -n# switch.
- Variant generation support temporarily disabled pending more general solution.
- Created a feature branch named **yunsaki-morphing-invoke** which adds experimental support for
  iteratively modifying the prompt and its parameters. Please see[Pull Request #86](https://github.com/lstein/stable-diffusion/pull/86)
  for a synopsis of how this works. Note that when this feature is eventually added to the main branch, it will may be modified
  significantly.

---

## v1.10 <small>(25 August 2022)</small>

- A barebones but fully functional interactive web server for online generation of txt2img and img2img.

---

## v1.09 <small>(24 August 2022)</small>

- A new -v option allows you to generate multiple variants of an initial image
  in img2img mode. (kudos to [Oceanswave](https://github.com/Oceanswave). [
  See this discussion in the PR for examples and details on use](https://github.com/lstein/stable-diffusion/pull/71#issuecomment-1226700810))
- Added ability to personalize text to image generation (kudos to [Oceanswave](https://github.com/Oceanswave) and [nicolai256](https://github.com/nicolai256))
- Enabled all of the samplers from k_diffusion

---

## v1.08 <small>(24 August 2022)</small>

- Escape single quotes on the invoke> command before trying to parse. This avoids
  parse errors.
- Removed instruction to get Python3.8 as first step in Windows install.
  Anaconda3 does it for you.
- Added bounds checks for numeric arguments that could cause crashes.
- Cleaned up the copyright and license agreement files.

---

## v1.07 <small>(23 August 2022)</small>

- Image filenames will now never fill gaps in the sequence, but will be assigned the
  next higher name in the chosen directory. This ensures that the alphabetic and chronological
  sort orders are the same.

---

## v1.06 <small>(23 August 2022)</small>

- Added weighted prompt support contributed by [xraxra](https://github.com/xraxra)
- Example of using weighted prompts to tweak a demonic figure contributed by [bmaltais](https://github.com/bmaltais)

---

## v1.05 <small>(22 August 2022 - after the drop)</small>

- Filenames now use the following formats:
  000010.95183149.png -- Two files produced by the same command (e.g. -n2),
  000010.26742632.png -- distinguished by a different seed.

  000011.455191342.01.png -- Two files produced by the same command using
  000011.455191342.02.png -- a batch size>1 (e.g. -b2). They have the same seed.

  000011.4160627868.grid#1-4.png -- a grid of four images (-g); the whole grid can
  be regenerated with the indicated key

- It should no longer be possible for one image to overwrite another
- You can use the "cd" and "pwd" commands at the invoke> prompt to set and retrieve
  the path of the output directory.

---

## v1.04 <small>(22 August 2022 - after the drop)</small>

- Updated README to reflect installation of the released weights.
- Suppressed very noisy and inconsequential warning when loading the frozen CLIP
  tokenizer.

---

## v1.03 <small>(22 August 2022)</small>

- The original txt2img and img2img scripts from the CompViz repository have been moved into
  a subfolder named "orig_scripts", to reduce confusion.

---

## v1.02 <small>(21 August 2022)</small>

- A copy of the prompt and all of its switches and options is now stored in the corresponding
  image in a tEXt metadata field named "Dream". You can read the prompt using scripts/images2prompt.py,
  or an image editor that allows you to explore the full metadata.
  **Please run "conda env update" to load the k_lms dependencies!!**

---

## v1.01 <small>(21 August 2022)</small>

- added k_lms sampling.
  **Please run "conda env update" to load the k_lms dependencies!!**
- use half precision arithmetic by default, resulting in faster execution and lower memory requirements
  Pass argument --full_precision to invoke.py to get slower but more accurate image generation

---

## Links

- **[Read Me](index.md)**

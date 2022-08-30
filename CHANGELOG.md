# **Changelog**

## v1.13 (in process)

- Supports a Google Colab notebook for a standalone server running on Google hardware [Arturo Mendivil](https://github.com/artmen1516)
- WebUI supports GFPGAN/ESRGAN facial reconstruction and upscaling [Kevin Gibbons](https://github.com/bakkot)
- WebUI supports incremental display of in-progress images during generation [Kevin Gibbons](https://github.com/bakkot)
- Output directory can be specified on the dream> command line.
- The grid was displaying duplicated images when not enough images to fill the final row [Muhammad Usama](https://github.com/SMUsamaShah)
- Can specify --grid on dream.py command line as the default.
- Miscellaneous internal bug and stability fixes.

---

## v1.12 (28 August 2022)

- Improved file handling, including ability to read prompts from standard input.
  (kudos to [Yunsaki](https://github.com/yunsaki)
- The web server is now integrated with the dream.py script. Invoke by adding --web to
  the dream.py command arguments.
- Face restoration and upscaling via GFPGAN and Real-ESGAN are now automatically
  enabled if the GFPGAN directory is located as a sibling to Stable Diffusion.
  VRAM requirements are modestly reduced. Thanks to both [Blessedcoolant](https://github.com/blessedcoolant) and
  [Oceanswave](https://github.com/oceanswave) for their work on this.
- You can now swap samplers on the dream> command line. [Blessedcoolant](https://github.com/blessedcoolant)

---

## v1.11 (26 August 2022)

- NEW FEATURE: Support upscaling and face enhancement using the GFPGAN module. (kudos to [Oceanswave](https://github.com/Oceanswave)
- You now can specify a seed of -1 to use the previous image's seed, -2 to use the seed for the image generated before that, etc.
  Seed memory only extends back to the previous command, but will work on all images generated with the -n# switch.
- Variant generation support temporarily disabled pending more general solution.
- Created a feature branch named **yunsaki-morphing-dream** which adds experimental support for
  iteratively modifying the prompt and its parameters. Please see[ Pull Request #86](https://github.com/lstein/stable-diffusion/pull/86)
  for a synopsis of how this works. Note that when this feature is eventually added to the main branch, it will may be modified
  significantly.

---

## v1.10 (25 August 2022)

- A barebones but fully functional interactive web server for online generation of txt2img and img2img.

---

## v1.09 (24 August 2022)

- A new -v option allows you to generate multiple variants of an initial image
  in img2img mode. (kudos to [Oceanswave](https://github.com/Oceanswave). [
  See this discussion in the PR for examples and details on use](https://github.com/lstein/stable-diffusion/pull/71#issuecomment-1226700810))
- Added ability to personalize text to image generation (kudos to [Oceanswave](https://github.com/Oceanswave) and [nicolai256](https://github.com/nicolai256))
- Enabled all of the samplers from k_diffusion

---

## v1.08 (24 August 2022)

- Escape single quotes on the dream> command before trying to parse. This avoids
  parse errors.
- Removed instruction to get Python3.8 as first step in Windows install.
  Anaconda3 does it for you.
- Added bounds checks for numeric arguments that could cause crashes.
- Cleaned up the copyright and license agreement files.

---

## v1.07 (23 August 2022)

- Image filenames will now never fill gaps in the sequence, but will be assigned the
  next higher name in the chosen directory. This ensures that the alphabetic and chronological
  sort orders are the same.

---

## v1.06 (23 August 2022)

- Added weighted prompt support contributed by [xraxra](https://github.com/xraxra)
- Example of using weighted prompts to tweak a demonic figure contributed by [bmaltais](https://github.com/bmaltais)

---

## v1.05 (22 August 2022 - after the drop)

- Filenames now use the following formats:
  000010.95183149.png -- Two files produced by the same command (e.g. -n2),
  000010.26742632.png -- distinguished by a different seed.

  000011.455191342.01.png -- Two files produced by the same command using
  000011.455191342.02.png -- a batch size>1 (e.g. -b2). They have the same seed.

  000011.4160627868.grid#1-4.png -- a grid of four images (-g); the whole grid can
  be regenerated with the indicated key

- It should no longer be possible for one image to overwrite another
- You can use the "cd" and "pwd" commands at the dream> prompt to set and retrieve
  the path of the output directory.

---

## v1.04 (22 August 2022 - after the drop)

- Updated README to reflect installation of the released weights.
- Suppressed very noisy and inconsequential warning when loading the frozen CLIP
  tokenizer.

---

## v1.03 (22 August 2022)

- The original txt2img and img2img scripts from the CompViz repository have been moved into
  a subfolder named "orig_scripts", to reduce confusion.

---

## v1.02 (21 August 2022)

- A copy of the prompt and all of its switches and options is now stored in the corresponding
  image in a tEXt metadata field named "Dream". You can read the prompt using scripts/images2prompt.py,
  or an image editor that allows you to explore the full metadata.
  **Please run "conda env update -f environment.yaml" to load the k_lms dependencies!!**

---

## v1.01 (21 August 2022)

- added k_lms sampling.
  **Please run "conda env update -f environment.yaml" to load the k_lms dependencies!!**
- use half precision arithmetic by default, resulting in faster execution and lower memory requirements
  Pass argument --full_precision to dream.py to get slower but more accurate image generation

---

## Links

- **[Read Me](readme.md)**

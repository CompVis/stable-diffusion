---
title: Home
---

<!--
  The Docs you find here (/docs/*) are built and deployed via mkdocs. If you want to do so from local it is pretty strait forward:

  ```bash
  pip install -r requirements-mkdocs.txt
  mkdocs serve -a localhost:8080
  ```
-->

<h1 align='center'><b>Stable Diffusion Dream Script</b></h1>

<p align='center'>
<img src="./assets/logo.png"/>
</p>

<p align="center">
    <img src="https://img.shields.io/github/last-commit/lstein/stable-diffusion?logo=Python&logoColor=green&style=for-the-badge" alt="last-commit"/>
    <img src="https://img.shields.io/github/stars/lstein/stable-diffusion?logo=GitHub&style=for-the-badge" alt="stars"/>
    <br>
    <img src="https://img.shields.io/github/issues/lstein/stable-diffusion?logo=GitHub&style=for-the-badge" alt="issues"/>
    <img src="https://img.shields.io/github/issues-pr/lstein/stable-diffusion?logo=GitHub&style=for-the-badge" alt="pull-requests"/>
</p>

This is a fork of [CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion), the open
source text-to-image generator. It provides a streamlined process with various new features and
options to aid the image generation process. It runs on Windows, Mac and Linux machines, and runs on
GPU cards with as little as 4 GB or RAM.

_Note: This fork is rapidly evolving. Please use the
[Issues](https://github.com/lstein/stable-diffusion/issues) tab to report bugs and make feature
requests. Be sure to use the provided templates. They will help aid diagnose issues faster._

## Installation

This fork is supported across multiple platforms. You can find individual installation instructions
below.

- [Linux](installation/INSTALL_LINUX.md)
- [Windows](installation/INSTALL_WINDOWS.md)
- [Macintosh](installation/INSTALL_MAC.md)

## Hardware Requirements

### System

You wil need one of the following:

- An NVIDIA-based graphics card with 4 GB or more VRAM memory.
- An Apple computer with an M1 chip.

### Memory

- At least 12 GB Main Memory RAM.

### Disk

- At least 6 GB of free disk space for the machine learning model, Python, and all its dependencies.

### Note

Precision is auto configured based on the device. If however you encounter
errors like 'expected type Float but found Half' or 'not implemented for Half'
you can try starting `dream.py` with the `--precision=float32` flag:

```bash
(ldm) ~/stable-diffusion$ python scripts/dream.py --precision=float32
```

## Features

### Major Features

- [Interactive Command Line Interface](features/CLI.md)
- [Image To Image](features/IMG2IMG.md)
- [Inpainting Support](features/INPAINTING.md)
- [GFPGAN and Real-ESRGAN Support](features/UPSCALE.md)
- [Seamless Tiling](features/OTHER.md#seamless-tiling)
- [Google Colab](features/OTHER.md#google-colab)
- [Web Server](features/WEB.md)
- [Reading Prompts From File](features/OTHER.md#reading-prompts-from-a-file)
- [Shortcut: Reusing Seeds](features/OTHER.md#shortcuts-reusing-seeds)
- [Weighted Prompts](features/OTHER.md#weighted-prompts)
- [Variations](features/VARIATIONS.md)
- [Personalizing Text-to-Image Generation](features/TEXTUAL_INVERSION.md)
- [Simplified API for text to image generation](features/OTHER.md#simplified-api)

### Other Features

- [Creating Transparent Regions for Inpainting](features/INPAINTING.md#creating-transparent-regions-for-inpainting)
- [Preload Models](features/OTHER.md#preload-models)

## Latest Changes

### vNEXT <small>(TODO 2022)</small>

  - Deprecated `--full_precision` / `-F`. Simply omit it and `dream.py` will auto
    configure. To switch away from auto use the new flag like `--precision=float32`.

### v1.14 <small>(11 September 2022)</small>

- Memory optimizations for small-RAM cards. 512x512 now possible on 4 GB GPUs.
- Full support for Apple hardware with M1 or M2 chips.
- Add "seamless mode" for circular tiling of image. Generates beautiful effects.
  ([prixt](https://github.com/prixt)).
- Inpainting support.
- Improved web server GUI.
- Lots of code and documentation cleanups.

### v1.13 <small>(3 September 2022</small>

- Support image variations (see [VARIATIONS](features/VARIATIONS.md)
  ([Kevin Gibbons](https://github.com/bakkot) and many contributors and reviewers)
- Supports a Google Colab notebook for a standalone server running on Google hardware
  [Arturo Mendivil](https://github.com/artmen1516)
- WebUI supports GFPGAN/ESRGAN facial reconstruction and upscaling
  [Kevin Gibbons](https://github.com/bakkot)
- WebUI supports incremental display of in-progress images during generation
  [Kevin Gibbons](https://github.com/bakkot)
- A new configuration file scheme that allows new models (including upcoming stable-diffusion-v1.5)
  to be added without altering the code. ([David Wager](https://github.com/maddavid12))
- Can specify --grid on dream.py command line as the default.
- Miscellaneous internal bug and stability fixes.
- Works on M1 Apple hardware.
- Multiple bug fixes.

For older changelogs, please visit the **[CHANGELOG](features/CHANGELOG.md)**.

## Troubleshooting

Please check out our **[Q&A](help/TROUBLESHOOT.md)** to get solutions for common installation
problems and other issues.

## Contributing

Anyone who wishes to contribute to this project, whether documentation, features, bug fixes, code
cleanup, testing, or code reviews, is very much encouraged to do so. If you are unfamiliar with how
to contribute to GitHub projects, here is a
[Getting Started Guide](https://opensource.com/article/19/7/create-pull-request-github).

A full set of contribution guidelines, along with templates, are in progress, but for now the most
important thing is to **make your pull request against the "development" branch**, and not against
"main". This will help keep public breakage to a minimum and will allow you to propose more radical
changes.

## Contributors

This fork is a combined effort of various people from across the world.
[Check out the list of all these amazing people](other/CONTRIBUTORS.md). We thank them for their
time, hard work and effort.

## Support

For support, please use this repository's GitHub Issues tracking service. Feel free to send me an
email if you use and like the script.

Original portions of the software are Copyright (c) 2020
[Lincoln D. Stein](https://github.com/lstein)

## Further Reading

Please see the original README for more information on this software and underlying algorithm,
located in the file [README-CompViz.md](other/README-CompViz.md).

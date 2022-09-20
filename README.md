<div align="center">

# Stable Diffusion Dream Script

![project logo](docs/assets/logo.png)

<p align='center'>
  <a href="https://discord.gg/ZmtBAhwWhy"><img src="docs/assets/join-us-on-discord-image.png"/></a>
</p>

# **Stable Diffusion Dream Script**
[![discord badge]][discord link]

[![latest release badge]][latest release link] [![github stars badge]][github stars link] [![github forks badge]][github forks link]

[![CI checks on main badge]][CI checks on main link] [![CI checks on dev badge]][CI checks on dev link] [![latest commit to dev badge]][latest commit to dev link]

[![github open issues badge]][github open issues link] [![github open prs badge]][github open prs link]

[CI checks on dev badge]: https://flat.badgen.net/github/checks/lstein/stable-diffusion/development?label=CI%20status%20on%20dev&cache=900&icon=github
[CI checks on dev link]: https://github.com/lstein/stable-diffusion/actions?query=branch%3Adevelopment
[CI checks on main badge]: https://flat.badgen.net/github/checks/lstein/stable-diffusion/main?label=CI%20status%20on%20main&cache=900&icon=github
[CI checks on main link]: https://github.com/lstein/stable-diffusion/actions/workflows/test-dream-conda.yml
[discord badge]: https://flat.badgen.net/discord/members/htRgbc7e?icon=discord
[discord link]: https://discord.com/invite/htRgbc7e
[github forks badge]: https://flat.badgen.net/github/forks/lstein/stable-diffusion?icon=github
[github forks link]: https://useful-forks.github.io/?repo=lstein%2Fstable-diffusion
[github open issues badge]: https://flat.badgen.net/github/open-issues/lstein/stable-diffusion?icon=github
[github open issues link]: https://github.com/lstein/stable-diffusion/issues?q=is%3Aissue+is%3Aopen
[github open prs badge]: https://flat.badgen.net/github/open-prs/lstein/stable-diffusion?icon=github
[github open prs link]: https://github.com/lstein/stable-diffusion/pulls?q=is%3Apr+is%3Aopen
[github stars badge]: https://flat.badgen.net/github/stars/lstein/stable-diffusion?icon=github
[github stars link]: https://github.com/lstein/stable-diffusion/stargazers
[latest commit to dev badge]: https://flat.badgen.net/github/last-commit/lstein/stable-diffusion/development?icon=github&color=yellow&label=last%20dev%20commit&cache=900
[latest commit to dev link]: https://github.com/lstein/stable-diffusion/commits/development
[latest release badge]: https://flat.badgen.net/github/release/lstein/stable-diffusion/development?icon=github
[latest release link]: https://github.com/lstein/stable-diffusion/releases
</div>

This is a fork of [CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion), the open
source text-to-image generator. It provides a streamlined process with various new features and
options to aid the image generation process. It runs on Windows, Mac and Linux machines, and runs on
GPU cards with as little as 4 GB or RAM.

_Note: This fork is rapidly evolving. Please use the
[Issues](https://github.com/lstein/stable-diffusion/issues) tab to report bugs and make feature
requests. Be sure to use the provided templates. They will help aid diagnose issues faster._

## Table of Contents

1. [Installation](#installation)
2. [Hardware Requirements](#hardware-requirements)
3. [Features](#features)
4. [Latest Changes](#latest-changes)
5. [Troubleshooting](#troubleshooting)
6. [Contributing](#contributing)
7. [Contributors](#contributors)
8. [Support](#support)
9. [Further Reading](#further-reading)

### Installation

This fork is supported across multiple platforms. You can find individual installation instructions
below.

- #### [Linux](docs/installation/INSTALL_LINUX.md)

- #### [Windows](docs/installation/INSTALL_WINDOWS.md)

- #### [Macintosh](docs/installation/INSTALL_MAC.md)

### Hardware Requirements

#### System

You wil need one of the following:

- An NVIDIA-based graphics card with 4 GB or more VRAM memory.
- An Apple computer with an M1 chip.

#### Memory

- At least 12 GB Main Memory RAM.

#### Disk

- At least 6 GB of free disk space for the machine learning model, Python, and all its dependencies.

#### Note

Precision is auto configured based on the device. If however you encounter
errors like 'expected type Float but found Half' or 'not implemented for Half'
you can try starting `dream.py` with the `--precision=float32` flag:

```bash
(ldm) ~/stable-diffusion$ python scripts/dream.py --precision=float32
```

### Features

#### Major Features

- [Interactive Command Line Interface](docs/features/CLI.md)
- [Image To Image](docs/features/IMG2IMG.md)
- [Inpainting Support](docs/features/INPAINTING.md)
- [GFPGAN and Real-ESRGAN Support](docs/features/UPSCALE.md)
- [Seamless Tiling](docs/features/OTHER.md#seamless-tiling)
- [Google Colab](docs/features/OTHER.md#google-colab)
- [Web Server](docs/features/WEB.md)
- [Reading Prompts From File](docs/features/PROMPTS.md#reading-prompts-from-a-file)
- [Shortcut: Reusing Seeds](docs/features/OTHER.md#shortcuts-reusing-seeds)
- [Weighted Prompts](docs/features/PROMPTS.md#weighted-prompts)
- [Negative/Unconditioned Prompts](docs/features/PROMPTS.md#negative-and-unconditioned-prompts)
- [Variations](docs/features/VARIATIONS.md)
- [Personalizing Text-to-Image Generation](docs/features/TEXTUAL_INVERSION.md)
- [Simplified API for text to image generation](docs/features/OTHER.md#simplified-api)

#### Other Features

- [Creating Transparent Regions for Inpainting](docs/features/INPAINTING.md#creating-transparent-regions-for-inpainting)
- [Preload Models](docs/features/OTHER.md#preload-models)

### Latest Changes

- vNEXT (TODO 2022)

  - Deprecated `--full_precision` / `-F`. Simply omit it and `dream.py` will auto
    configure. To switch away from auto use the new flag like `--precision=float32`.

- v1.14 (11 September 2022)

  - Memory optimizations for small-RAM cards. 512x512 now possible on 4 GB GPUs.
  - Full support for Apple hardware with M1 or M2 chips.
  - Add "seamless mode" for circular tiling of image. Generates beautiful effects.
    ([prixt](https://github.com/prixt)).
  - Inpainting support.
  - Improved web server GUI.
  - Lots of code and documentation cleanups.

- v1.13 (3 September 2022

  - Support image variations (see [VARIATIONS](docs/features/VARIATIONS.md)
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
  - Can specify --grid on dream.py command line as the default.
  - Miscellaneous internal bug and stability fixes.
  - Works on M1 Apple hardware.
  - Multiple bug fixes.

For older changelogs, please visit the **[CHANGELOG](docs/features/CHANGELOG.md)**.

### Troubleshooting

Please check out our **[Q&A](docs/help/TROUBLESHOOT.md)** to get solutions for common installation
problems and other issues.

### Contributing

Anyone who wishes to contribute to this project, whether documentation, features, bug fixes, code
cleanup, testing, or code reviews, is very much encouraged to do so. If you are unfamiliar with how
to contribute to GitHub projects, here is a
[Getting Started Guide](https://opensource.com/article/19/7/create-pull-request-github).

A full set of contribution guidelines, along with templates, are in progress, but for now the most
important thing is to **make your pull request against the "development" branch**, and not against
"main". This will help keep public breakage to a minimum and will allow you to propose more radical
changes.

### Contributors

This fork is a combined effort of various people from across the world.
[Check out the list of all these amazing people](docs/other/CONTRIBUTORS.md). We thank them for
their time, hard work and effort.

### Support

For support, please use this repository's GitHub Issues tracking service. Feel free to send me an
email if you use and like the script.

Original portions of the software are Copyright (c) 2020
[Lincoln D. Stein](https://github.com/lstein)

### Further Reading

Please see the original README for more information on this software and underlying algorithm,
located in the file [README-CompViz.md](docs/other/README-CompViz.md).

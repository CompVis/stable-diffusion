# **Interactive Command-Line Interface**

The `dream.py` script, located in `scripts/dream.py`, provides an interactive interface to image generation similar to the "dream mothership" bot that Stable AI provided on its Discord server.

Unlike the txt2img.py and img2img.py scripts provided in the original CompViz/stable-diffusion source code repository, the time-consuming initialization of the AI model initialization only happens once. After that image generation
from the command-line interface is very fast.

The script uses the readline library to allow for in-line editing, command history (up and down arrows), autocompletion, and more. To help keep track of which prompts generated which images, the script writes a log file of image names and prompts to the selected output directory.

In addition, as of version 1.02, it also writes the prompt into the PNG file's metadata where it can be retrieved using scripts/images2prompt.py

The script is confirmed to work on Linux, Windows and Mac systems.

_Note:_ This script runs from the command-line or can be used as a Web application. The Web GUI is currently rudimentary, but a much better replacement is on its way.

```
(ldm) ~/stable-diffusion$ python3 ./scripts/dream.py
* Initializing, be patient...
Loading model from models/ldm/text2img-large/model.ckpt
(...more initialization messages...)

* Initialization done! Awaiting your command...
dream> ashley judd riding a camel -n2 -s150
Outputs:
   outputs/img-samples/00009.png: "ashley judd riding a camel" -n2 -s150 -S 416354203
   outputs/img-samples/00010.png: "ashley judd riding a camel" -n2 -s150 -S 1362479620

dream> "there's a fly in my soup" -n6 -g
    outputs/img-samples/00011.png: "there's a fly in my soup" -n6 -g -S 2685670268
    seeds for individual rows: [2685670268, 1216708065, 2335773498, 822223658, 714542046, 3395302430]
dream> q

# this shows how to retrieve the prompt stored in the saved image's metadata
(ldm) ~/stable-diffusion$ python ./scripts/images2prompt.py outputs/img_samples/*.png
00009.png: "ashley judd riding a camel" -s150 -S 416354203
00010.png: "ashley judd riding a camel" -s150 -S 1362479620
00011.png: "there's a fly in my soup" -n6 -g -S 2685670268
```

<p align='center'>
<img src="../assets/dream-py-demo.png"/>
</p>

The `dream>` prompt's arguments are pretty much identical to those used in the Discord bot, except you don't need to type "!dream" (it doesn't
hurt if you do). A significant change is that creation of individual images is now the default unless --grid (-g) is given.

For backward compatibility, the -i switch is recognized. For command-line help type -h (or --help) at the dream> prompt.

The script itself also recognizes a series of command-line switches that will change important global defaults, such as the directory for
image outputs and the location of the model weight files.

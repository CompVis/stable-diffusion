# Stable Diffusion Dream Script

This is a fork of CompVis/stable-diffusion, the wonderful open source
text-to-image generator.

The original has been modified in several ways:

## Interactive command-line interface similar to the Discord bot

The *dream.py* script, located in scripts/dream.py, 
provides an interactive interface to image generation similar to
the "dream mothership" bot that Stable AI provided on its Discord
server. Unlike the txt2img.py and img2img.py scripts provided in the
original CompViz/stable-diffusion source code repository, the
time-consuming initialization of the AI model
initialization only happens once. After that image generation 
from the command-line interface is very fast.

The script uses the readline library to allow for in-line editing,
command history (up and down arrows), autocompletion, and more. To help
keep track of which prompts generated which images, the script writes a
log file of image names and prompts to the selected output directory.
In addition, as of version 1.02, it also writes the prompt into the PNG
file's metadata where it can be retrieved using scripts/images2prompt.py

The script is confirmed to work on Linux and Windows systems. It should
work on MacOSX as well, but this is not confirmed. Note that this script
runs from the command-line (CMD or Terminal window), and does not have a GUI.

~~~~
(ldm) ~/stable-diffusion$ python3 ./scripts/dream.py
* Initializing, be patient...
Loading model from models/ldm/text2img-large/model.ckpt
LatentDiffusion: Running in eps-prediction mode
DiffusionWrapper has 872.30 M params.
making attention of type 'vanilla' with 512 in_channels
Working with z of shape (1, 4, 32, 32) = 4096 dimensions.
making attention of type 'vanilla' with 512 in_channels
Loading Bert tokenizer from "models/bert"
setting sampler to plms

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
(ldm) ~/stable-diffusion$ python3 ./scripts/images2prompt.py outputs/img_samples/*.png
00009.png: "ashley judd riding a camel" -s150 -S 416354203
00010.png: "ashley judd riding a camel" -s150 -S 1362479620
00011.png: "there's a fly in my soup" -n6 -g -S 2685670268
~~~~

The dream> prompt's arguments are pretty much identical to those used
in the Discord bot, except you don't need to type "!dream" (it doesn't
hurt if you do). A significant change is that creation of individual
images is now the default unless --grid (-g) is given. For backward
compatibility, the -i switch is recognized.  For command-line help
type -h (or --help) at the dream> prompt.

The script itself also recognizes a series of command-line switches
that will change important global defaults, such as the directory for
image outputs and the location of the model weight files.

## Image-to-Image

This script also provides an img2img feature that lets you seed your
creations with a drawing or photo. This is a really cool feature that tells
stable diffusion to build the prompt on top of the image you provide, preserving
the original's basic shape and layout. To use it, provide the --init_img 
option as shown here:

~~~~
dream> "waterfall and rainbow" --init_img=./init-images/crude_drawing.png --strength=0.5 -s100 -n4
~~~~

The --init_img (-I) option gives the path to the seed picture. --strength (-f) controls how much
the original will be modified, ranging from 0.0 (keep the original intact), to 1.0 (ignore the original
completely). The default is 0.75, and ranges from 0.25-0.75 give interesting results.

You may also pass a -v<count> option to generate count variants on the original image. This is done by
passing the first generated image back into img2img the requested number of times. It generates interesting
variants.

## Barebones Web Server

As of version 1.10, this distribution comes with a bare bones web server (see screenshot). To use it,
run the command:

~~~~
(ldm) ~/stable-diffusion$ python3 scripts/dream_web.py
~~~~

You can then connect to the server by pointing your web browser at
http://localhost:9090, or to the network name or IP address of the server.

Kudos to [Tesseract Cat](https://github.com/TesseractCat) for
contributing this code.

![Dream Web Server](static/dream_web_server.png)

## Weighted Prompts

You may weight different sections of the prompt to tell the sampler to attach different levels of
priority to them, by adding :(number) to the end of the section you wish to up- or downweight.
For example consider this prompt:

~~~~
    tabby cat:0.25 white duck:0.75 hybrid
~~~~

This will tell the sampler to invest 25% of its effort on the tabby
cat aspect of the image and 75% on the white duck aspect
(surprisingly, this example actually works). The prompt weights can
use any combination of integers and floating point numbers, and they
do not need to add up to 1.

## Personalizing Text-to-Image Generation

You may personalize the generated images to provide your own styles or objects by training a new LDM checkpoint
and introducing a new vocabulary to the fixed model.

To train, prepare a folder that contains images sized at 512x512 and execute the following:

~~~~
# As the default backend is not available on Windows, if you're using that platform, execute SET PL_TORCH_DISTRIBUTED_BACKEND=gloo
(ldm) ~/stable-diffusion$ python3 ./main.py --base ./configs/stable-diffusion/v1-finetune.yaml \
                                            -t \
                                            --actual_resume ./models/ldm/stable-diffusion-v1/model.ckpt \
                                            -n my_cat \
                                            --gpus 0, \
                                            --data_root D:/textual-inversion/my_cat \
                                            --init_word 'cat'
~~~~

During the training process, files will be created in /logs/[project][time][project]/
where you can see the process.

conditioning* contains the training prompts
inputs, reconstruction the input images for the training epoch
samples, samples scaled for a sample of the prompt and one with the init word provided 

On a RTX3090, the process for SD will take ~1h @1.6 iterations/sec.

Note: According to the associated paper, the optimal number of images
is 3-5. Your model may not converge if you use more images than that.

Training will run indefinately, but you may wish to stop it before the
heat death of the universe, when you find a low loss epoch or around
~5000 iterations.

Once the model is trained, specify the trained .pt file when starting
dream using

~~~~
(ldm) ~/stable-diffusion$ python3 ./scripts/dream.py --embedding_path /path/to/embedding.pt --full_precision
~~~~

Then, to utilize your subject at the dream prompt

~~~
dream> "a photo of *"
~~~

this also works with image2image
~~~~
dream> "waterfall and rainbow in the style of *" --init_img=./init-images/crude_drawing.png --strength=0.5 -s100 -n4
~~~~

It's also possible to train multiple tokens (modify the placeholder string in configs/stable-diffusion/v1-finetune.yaml) and combine LDM checkpoints using:

~~~~
(ldm) ~/stable-diffusion$ python3 ./scripts/merge_embeddings.py \
                                            --manager_ckpts /path/to/first/embedding.pt /path/to/second/embedding.pt [...] \
                                            --output_path /path/to/output/embedding.pt
~~~~

Credit goes to @rinongal and the repository located at
https://github.com/rinongal/textual_inversion Please see the
repository and associated paper for details and limitations.

## Changes

* v1.09 (24 August 2022)
   * A barebone web server for interactive online generation of txt2img and img2img.
   * A new -v option allows you to generate multiple variants of an initial image
     in img2img mode. (kudos to [Oceanswave](https://github.com/Oceanswave). [See this discussion in the PR for examples and details on use](https://github.com/lstein/stable-diffusion/pull/71#issuecomment-1226700810))
   * Added ability to personalize text to image generation (kudos to [Oceanswave](https://github.com/Oceanswave) and [nicolai256](https://github.com/nicolai256))
   * Enabled all of the samplers from k_diffusion
   
* v1.08 (24 August 2022)
   * Escape single quotes on the dream> command before trying to parse. This avoids
     parse errors.
   * Removed instruction to get Python3.8 as first step in Windows install.
     Anaconda3 does it for you.
   * Added bounds checks for numeric arguments that could cause crashes.
   * Cleaned up the copyright and license agreement files.

* v1.07 (23 August 2022)
   * Image filenames will now never fill gaps in the sequence, but will be assigned the
     next higher name in the chosen directory. This ensures that the alphabetic and chronological
     sort orders are the same.

* v1.06 (23 August 2022)
   * Added weighted prompt support contributed by [xraxra](https://github.com/xraxra)
   * Example of using weighted prompts to tweak a demonic figure contributed by [bmaltais](https://github.com/bmaltais)

* v1.05 (22 August 2022 - after the drop)
   * Filenames now use the following formats:
       000010.95183149.png      -- Two files produced by the same command (e.g. -n2),
       000010.26742632.png      -- distinguished by a different seed.

       000011.455191342.01.png  -- Two files produced by the same command using
       000011.455191342.02.png  -- a batch size>1 (e.g. -b2). They have the same seed.

       000011.4160627868.grid#1-4.png  -- a grid of four images (-g); the whole grid can
                                          be regenerated with the indicated key

    * It should no longer be possible for one image to overwrite another
    * You can use the "cd" and "pwd" commands at the dream> prompt to set and retrieve
      the path of the output directory.
     
* v1.04 (22 August 2022 - after the drop)
   * Updated README to reflect installation of the released weights.
   * Suppressed very noisy and inconsequential warning when loading the frozen CLIP
   tokenizer.

* v1.03 (22 August 2022)
   * The original txt2img and img2img scripts from the CompViz repository have been moved into
     a subfolder named "orig_scripts", to reduce confusion.
   
* v1.02 (21 August 2022)
    * A copy of the prompt and all of its switches and options is now stored in the corresponding
    image in a tEXt metadata field named "Dream". You can read the prompt using scripts/images2prompt.py,
    or an image editor that allows you to explore the full metadata.
        **Please run "conda env update -f environment.yaml" to load the k_lms dependencies!!**

* v1.01 (21 August 2022)
    * added k_lms sampling. 
      **Please run "conda env update -f environment.yaml" to load the k_lms dependencies!!**
    * use half precision arithmetic by default, resulting in faster execution and lower memory requirements
    Pass argument --full_precision to dream.py to get slower but more accurate image generation


## Installation

There are separate installation walkthroughs for [Linux/Mac](#linuxmac) and [Windows](#windows).

### Linux/Mac

1. You will need to install the following prerequisites if they are not already available. Use your
operating system's preferred installer
* Python (version 3.8.5 recommended; higher may work)
* git

2. Install the Python Anaconda environment manager using pip3.
```
~$ pip3 install anaconda
```
After installing anaconda, you should log out of your system and log back in. If the installation
worked, your command prompt will be prefixed by the name of the current anaconda environment, "(base)".

3. Copy the stable-diffusion source code from GitHub:
```
(base) ~$ git clone https://github.com/lstein/stable-diffusion.git
```
This will create stable-diffusion folder where you will follow the rest of the steps.

4. Enter the newly-created stable-diffusion folder. From this step forward make sure that you are working in the stable-diffusion directory!
```
(base) ~$ cd stable-diffusion
(base) ~/stable-diffusion$
```
5. Use anaconda to copy necessary python packages, create a new python environment named "ldm",
and activate the environment. 
```
(base) ~/stable-diffusion$ conda env create -f environment.yaml
(base) ~/stable-diffusion$ conda activate ldm
(ldm) ~/stable-diffusion$
```
After these steps, your command prompt will be prefixed by "(ldm)" as shown above.

6. Load a couple of small machine-learning models required by stable diffusion:
```
(ldm) ~/stable-diffusion$ python3 scripts/preload_models.py
```

Note that this step is necessary because I modified the original
just-in-time model loading scheme to allow the script to work on GPU
machines that are not internet connected. See [Workaround for machines with limited internet connectivity](#workaround-for-machines-with-limited-internet-connectivity)

7. Now you need to install the weights for the stable diffusion model.

For running with the released weights, you will first need to set up an acount with Hugging Face (https://huggingface.co).
Use your credentials to log in, and then point your browser at https://huggingface.co/CompVis/stable-diffusion-v-1-4-original.
You may be asked to sign a license agreement at this point.

Click on "Files and versions" near the top of the page, and then click on the file named "sd-v1-4.ckpt". You'll be taken
to a page that prompts you to click the "download" link. Save the file somewhere safe on your local machine.

Now run the following commands from within the stable-diffusion directory. This will create a symbolic
link from the stable-diffusion model.ckpt file, to the true location of the sd-v1-4.ckpt file.
   
```
(ldm) ~/stable-diffusion$ mkdir -p models/ldm/stable-diffusion-v1
(ldm) ~/stable-diffusion$ ln -sf /path/to/sd-v1-4.ckpt models/ldm/stable-diffusion-v1/model.ckpt
```

8. Start generating images!
```
# for the pre-release weights use the -l or --liaon400m switch
(ldm) ~/stable-diffusion$ python3 scripts/dream.py -l

# for the post-release weights do not use the switch
(ldm) ~/stable-diffusion$ python3 scripts/dream.py

# for additional configuration switches and arguments, use -h or --help
(ldm) ~/stable-diffusion$ python3 scripts/dream.py -h
```
9. Subsequently, to relaunch the script, be sure to run "conda activate ldm" (step 5, second command), enter the "stable-diffusion" 
directory, and then launch the dream script (step 8). If you forget to activate the ldm environment, the script will fail with multiple ModuleNotFound errors.

#### Updating to newer versions of the script

This distribution is changing rapidly. If you used the "git clone" method (step 5) to download the stable-diffusion directory, then to update to the latest and greatest version, launch the Anaconda window, enter "stable-diffusion", and type:
```
(ldm) ~/stable-diffusion$ git pull
```
This will bring your local copy into sync with the remote one.

### Windows

1. Install Anaconda3 (miniconda3 version) from here: https://docs.anaconda.com/anaconda/install/windows/

2. Install Git from here: https://git-scm.com/download/win

3. Launch Anaconda from the Windows Start menu. This will bring up a command window. Type all the remaining commands in this window.

4. Run the command:
```
git clone https://github.com/lstein/stable-diffusion.git
```
This will create stable-diffusion folder where you will follow the rest of the steps.

5. Enter the newly-created stable-diffusion folder. From this step forward make sure that you are working in the stable-diffusion directory!
```
cd stable-diffusion
```

6. Run the following two commands:
```
conda env create -f environment.yaml    (step 6a)
conda activate ldm                      (step 6b)
```
This will install all python requirements and activate the "ldm" environment which sets PATH and other environment variables properly.

7. Run the command:
```
python scripts\preload_models.py
```

This installs several machine learning models that stable diffusion
requires. (Note that this step is required. I created it because some people
are using GPU systems that are behind a firewall and the models can't be
downloaded just-in-time)

8. Now you need to install the weights for the big stable diffusion model.

For running with the released weights, you will first need to set up
an acount with Hugging Face (https://huggingface.co).  Use your
credentials to log in, and then point your browser at
https://huggingface.co/CompVis/stable-diffusion-v-1-4-original.  You
may be asked to sign a license agreement at this point.

Click on "Files and versions" near the top of the page, and then click
on the file named "sd-v1-4.ckpt". You'll be taken to a page that
prompts you to click the "download" link. Now save the file somewhere
safe on your local machine.  The weight file is >4 GB in size, so
downloading may take a while.

Now run the following commands from **within the stable-diffusion
directory** to copy the weights file to the right place:
   
```
mkdir -p models\ldm\stable-diffusion-v1
copy C:\path\to\sd-v1-4.ckpt models\ldm\stable-diffusion-v1\model.ckpt
```
Please replace "C:\path\to\sd-v1.4.ckpt" with the correct path to wherever
you stashed this file. If you prefer not to copy or move the .ckpt file, 
you may instead create a shortcut to it from within
"models\ldm\stable-diffusion-v1\".

9. Start generating images!
```
# for the pre-release weights
python scripts\dream.py -l

# for the post-release weights
python scripts\dream.py
```
10. Subsequently, to relaunch the script, first activate the Anaconda command window (step 3), enter the stable-diffusion directory (step 5, "cd \path\to\stable-diffusion"), run "conda activate ldm" (step 6b), and then launch the dream script (step 9).

#### Updating to newer versions of the script

This distribution is changing rapidly. If you used the "git clone" method (step 5) to download the stable-diffusion directory, then to update to the latest and greatest version, launch the Anaconda window, enter "stable-diffusion", and type:
```
git pull
```
This will bring your local copy into sync with the remote one.

## Simplified API for text to image generation

For programmers who wish to incorporate stable-diffusion into other
products, this repository includes a simplified API for text to image generation, which
lets you create images from a prompt in just three lines of code:

~~~~
from ldm.simplet2i import T2I
model   = T2I()
outputs = model.txt2img("a unicorn in manhattan")
~~~~

Outputs is a list of lists in the format [[filename1,seed1],[filename2,seed2]...]
Please see ldm/simplet2i.py for more information.


## Workaround for machines with limited internet connectivity

My development machine is a GPU node in a high-performance compute
cluster which has no connection to the internet. During model
initialization, stable-diffusion tries to download the Bert tokenizer
and a file needed by the kornia library.  This obviously didn't work
for me.

To work around this, I have modified ldm/modules/encoders/modules.py
to look for locally cached Bert files rather than attempting to
download them. For this to work, you must run
"scripts/preload_models.py" once from an internet-connected machine
prior to running the code on an isolated one. This assumes that both
machines share a common network-mounted filesystem with a common
.cache directory.

~~~~
(ldm) ~/stable-diffusion$ python3 ./scripts/preload_models.py
preloading bert tokenizer...
Downloading: 100%|██████████████████████████████████| 28.0/28.0 [00:00<00:00, 49.3kB/s]
Downloading: 100%|██████████████████████████████████| 226k/226k [00:00<00:00, 2.79MB/s]
Downloading: 100%|██████████████████████████████████| 455k/455k [00:00<00:00, 4.36MB/s]
Downloading: 100%|██████████████████████████████████| 570/570 [00:00<00:00, 477kB/s]
...success
preloading kornia requirements...
Downloading: "https://github.com/DagnyT/hardnet/raw/master/pretrained/train_liberty_with_aug/checkpoint_liberty_with_aug.pth" to /u/lstein/.cache/torch/hub/checkpoints/checkpoint_liberty_with_aug.pth
100%|███████████████████████████████████████████████| 5.10M/5.10M [00:00<00:00, 101MB/s]
...success
~~~~

If you don't need this change and want to download the files just in
time, copy over the file ldm/modules/encoders/modules.py from the
CompVis/stable-diffusion repository. Or you can run preload_models.py
on the target machine.

## Support

For support,
please use this repository's GitHub Issues tracking service. Feel free
to send me an email if you use and like the script.

*Original Author:* Lincoln D. Stein <lincoln.stein@gmail.com>

*Contributions by:* 
[Peter Kowalczyk](https://github.com/slix), [Henry Harrison](https://github.com/hwharrison),
[xraxra](https://github.com/xraxra), [bmaltais](https://github.com/bmaltais), [Sean McLellan](https://github.com/Oceanswave),
[nicolai256](https://github.com/nicolai256), [Benjamin Warner](https://github.com/warner-benjamin),
[tildebyte](https://github.com/tildebyte),
and [Tesseract Cat](https://github.com/TesseractCat)


Original portions of the software are Copyright (c) 2020 Lincoln D. Stein (https://github.com/lstein)

#Further Reading

Please see the original README for more information on this software
and underlying algorithm, located in the file README-CompViz.md.

print("Importing libraries and attempting to update files. This may take one or more minutes.")

# Import core libraries
import os, re, time, sys, asyncio, ctypes, math, threading, platform, subprocess
import torch
import scipy
import numpy as np
from random import randint
from omegaconf import OmegaConf
from PIL import Image
from itertools import islice, product
from einops import rearrange, repeat
from pytorch_lightning import seed_everything
from contextlib import nullcontext
from typing import Optional
from safetensors.torch import load_file
from cryptography.fernet import Fernet

# Import built libraries
from ldm.util import instantiate_from_config
from optimUtils import split_weighted_subprompts
from autoencoder.pixelvae import load_pixelvae_model
from lora.lora import apply_lora, assign_lora_names_to_compvis_modules, load_lora, register_lora_for_inference, remove_lora_for_inference
import hitherdither
import tomesd

# Import PyTorch functions
from torch import autocast
from torch import Tensor
from torch.nn import functional as F
from torch.nn.modules.utils import _pair

# Import logging libraries
import traceback, warnings
import logging as pylog
from transformers import logging

# Import websocket tools
import requests
from websockets import serve, connect
from io import BytesIO

# Import console management libraries
from rich import print as rprint
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    try:
        import pygetwindow as gw
    except:
        rprint(f"[#ab333d]Pygetwindow could not be loaded. This will limit some cosmetic functionality.")
from colorama import just_fix_windows_console
import playsound

system = platform.system()

if system == "Windows":
    # Fix windows console for color codes
    just_fix_windows_console()

    # Patch existing console to remove interactivity
    kernel32 = ctypes.windll.kernel32
    kernel32.SetConsoleMode(kernel32.GetStdHandle(-10), 128)

log = pylog.getLogger("pytorch_lightning")
log.propagate = False
log.setLevel(pylog.ERROR)
logging.set_verbosity_error()

global model
global modelCS
global modelFS
global modelPV
global running

global sounds
sounds = False

expectedVersion = "8.0.0"

# For testing only, limits memory usage to "maxMemory"
maxMemory = 4
if False:
    cardMemory = torch.cuda.get_device_properties("cuda").total_memory / 1073741824
    usedMemory = cardMemory - (torch.cuda.mem_get_info()[0] / 1073741824)

    fractionalMaxMemory = (maxMemory - (usedMemory+0.3)) / cardMemory
    print(usedMemory)
    print(cardMemory)
    print(maxMemory)
    print(cardMemory*fractionalMaxMemory)

    torch.cuda.set_per_process_memory_fraction(fractionalMaxMemory)

global timeout
global loaded
loaded = ""

def audioThread(file):
    try:
        absoluteFile = os.path.abspath(f"../sounds/{file}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            playsound.playsound(absoluteFile)
    except:
        pass

def play(file):
    global sounds
    if sounds == "true":
        try:
            threading.Thread(target=audioThread, args=(file,), daemon=True).start()
        except:
            pass

def patch_conv(**patch):
    # Patch the Conv2d class with a custom __init__ method
    cls = torch.nn.Conv2d
    init = cls.__init__

    def __init__(self, *args, **kwargs):
        # Call the original init method and apply the patch arguments
        return init(self, *args, **kwargs, **patch)
    
    cls.__init__ = __init__

def patch_conv_asymmetric(model, x, y):
    # Patch Conv2d layers in the given model for asymmetric padding
    for layer in flatten(model):
        if type(layer) == torch.nn.Conv2d:
            # Set padding mode based on x and y arguments
            layer.padding_modeX = 'circular' if x else 'constant'
            layer.padding_modeY = 'circular' if y else 'constant'

            # Compute padding values based on reversed padding repeated twice
            layer.paddingX = (layer._reversed_padding_repeated_twice[0], layer._reversed_padding_repeated_twice[1], 0, 0)
            layer.paddingY = (0, 0, layer._reversed_padding_repeated_twice[2], layer._reversed_padding_repeated_twice[3])

            # Patch the _conv_forward method with a replacement function
            layer._conv_forward = __replacementConv2DConvForward.__get__(layer, torch.nn.Conv2d)

def restoreConv2DMethods(model):
        # Restore original _conv_forward method for Conv2d layers in the model
        for layer in flatten(model):
            if type(layer) == torch.nn.Conv2d:
                layer._conv_forward = torch.nn.Conv2d._conv_forward.__get__(layer, torch.nn.Conv2d)

def __replacementConv2DConvForward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
    # Replacement function for Conv2d's _conv_forward method
    working = F.pad(input, self.paddingX, mode=self.padding_modeX)
    working = F.pad(working, self.paddingY, mode=self.padding_modeY)
    return F.conv2d(working, weight, bias, self.stride, _pair(0), self.dilation, self.groups)

def patch_tiling(tilingX, tilingY, model, modelFS, modelPV):
    # Convert tilingX and tilingY to boolean values
    X = bool(tilingX == "true")
    Y = bool(tilingY == "true")

    # Patch Conv2d layers in the given models for asymmetric padding
    patch_conv_asymmetric(model, X, Y)
    patch_conv_asymmetric(modelFS, X, Y)
    patch_conv_asymmetric(modelPV.model, X, Y)

    if X or Y:
        # Print a message indicating the direction(s) patched for tiling
        rprint("[#494b9b]Patched for tiling in the [#48a971]" + "X" * X + "[#494b9b] and [#48a971]" * (X and Y) + "Y" * Y + "[#494b9b] direction" + "s" * (X and Y))

    return model, modelFS, modelPV

def chunk(it, size):
    # Create an iterator from the input iterable
    it = iter(it)

    # Return an iterator that yields tuples of the specified size
    return iter(lambda: tuple(islice(it, size)), ())

def searchString(string, *args):
    out = []

    # Iterate over the range of arguments, excluding the last one
    for x in range(len(args)-1):
        # Perform a regex search in the string using the current and next argument as lookaround patterns
        # Append the matched substring to the output list
        try:
            out.append(re.search(f"(?<={{{args[x]}}}).*(?={{{args[x+1]}}})", string).group())
        except:
            if args[x] not in string:
                rprint(f"\n[#ab333d]Could not find: {args[x]}")

    return out

def climage(image, alignment, *args):

    # Get console bounds with a small margin - better safe than sorry
    twidth, theight = os.get_terminal_size().columns-1, (os.get_terminal_size().lines-1)*2

    # Set up variables
    image = image.convert('RGBA')
    iwidth, iheight = min(twidth, image.width), min(theight, image.height)
    line = []
    lines = []

    # Alignment stuff

    margin = 0
    if alignment == "centered":
        margin = int((twidth/2)-(iwidth/2))
    elif alignment == "right":
        margin = int(twidth-iwidth)
    elif alignment == "manual":
        margin = args[0]
    
    # Loop over the height of the image / 2 (because 2 pixels = 1 text character)
    for y2 in range(int(iheight/2)):

        # Add default colors to the start of the line
        line = [" "*margin]

        # Loop over width
        for x in range(iwidth):

            # Get the color for the upper and lower half of the text character
            r, g, b, a = image.getpixel((x, (y2*2)))
            r2, g2, b2, a2 = image.getpixel((x, (y2*2)+1))

            # Convert to hex colors for Rich to use
            rgb, rgb2 = '#{:02x}{:02x}{:02x}'.format(r, g, b), '#{:02x}{:02x}{:02x}'.format(r2, g2, b2)

            # Lookup table because I was bored
            colorCodes = [f"{rgb2} on {rgb}", f"{rgb2}", f"{rgb}", "nothing", f"{rgb}"]
            # ~It just works~
            maping = int(a < 200)+(int(a2 < 200)*2)+(int(rgb == rgb2 and a + a2 > 400)*4)
            color = colorCodes[maping]
            
            # Set text characters, nothing, full block, half block. Half block + background color = 2 pixels
            if a < 200 and a2 < 200:
                line.append(f" ")
            elif rgb == rgb2:
                line.append(f"[{color}]█[/{color}]")
            else:
                if maping == 2:
                    line.append(f"[{color}]▀[/{color}]")
                else:
                    line.append(f"[{color}]▄[/{color}]")
        
        # Add default colors to the end of the line
        lines.append("".join(line) + "")
    return "\n".join(lines)

def clbar(iterable, name = "", printEnd = "\r", position = "", unit = "it", disable = False, prefixwidth = 1, suffixwidth = 1, total = 0):

    # Console manipulation stuff
    def up(lines = 1):
        for _ in range(lines):
            sys.stdout.write('\x1b[1A')
            sys.stdout.flush()

    def down(lines = 1):
        for _ in range(lines):
            sys.stdout.write('\n')
            sys.stdout.flush()

    # Allow the complete disabling of the progress bar
    if not disable:
        # Positions the bar correctly
        down(int(position == "last")*2)
        up(int(position == "first")*3)
        
        # Set up variables
        if total > 0:
            #iterable = iterable[0:total]
            pass
        else:
            total = max(1, len(iterable))
        name = f"{name}"
        speed = f" {total}/{total} at 100.00 {unit}/s "
        prediction = f" 00:00 < 00:00 "
        prefix = max(len(name), len("100%"), prefixwidth)
        suffix = max(len(speed), len(prediction), suffixwidth)
        barwidth = os.get_terminal_size().columns-(suffix+prefix+2)

        # Prints the progress bar
        def printProgressBar (iteration, delay):

            # Define progress bar graphic
            line1 = ["[#494b9b on #3b1725]▄[/#494b9b on #3b1725]", 
                    "[#c4f129 on #494b9b]▄[/#c4f129 on #494b9b]" * int(int(barwidth * iteration // total) > 0), 
                    "[#ffffff on #494b9b]▄[/#ffffff on #494b9b]" * max(0, int(barwidth * iteration // total)-2),
                    "[#c4f129 on #494b9b]▄[/#c4f129 on #494b9b]" * int(int(barwidth * iteration // total) > 1),
                    "[#3b1725 on #494b9b]▄[/#3b1725 on #494b9b]" * max(0, barwidth-int(barwidth * iteration // total)),
                    "[#494b9b on #3b1725]▄[/#494b9b on #3b1725]"]
            line2 = ["[#3b1725 on #494b9b]▄[/#3b1725 on #494b9b]", 
                    "[#494b9b on #48a971]▄[/#494b9b on #48a971]" * int(int(barwidth * iteration // total) > 0), 
                    "[#494b9b on #c4f129]▄[/#494b9b on #c4f129]" * max(0, int(barwidth * iteration // total)-2),
                    "[#494b9b on #48a971]▄[/#494b9b on #48a971]" * int(int(barwidth * iteration // total) > 1),
                    "[#494b9b on #3b1725]▄[/#494b9b on #3b1725]" * max(0, barwidth-int(barwidth * iteration // total)),
                    "[#3b1725 on #494b9b]▄[/#3b1725 on #494b9b]"]

            percent = ("{0:.0f}").format(100 * (iteration / float(total)))

            # Avoid predicting speed until there's enough data
            if len(delay) >= 1:
                delay.append(time.time()-delay[-1])
                del delay [-2]

            # Fancy color stuff and formating
            if iteration == 0:
                speedColor = "[#48a971]"
                measure = f"... {unit}/s"
                passed = f"00:00"
                remaining = f"??:??"
            else:
                if np.mean(delay) <= 1:
                    measure = f"{round(1/max(0.01, np.mean(delay)), 2)} {unit}/s"
                else:
                    measure = f"{round(np.mean(delay), 2)} s/{unit}"

                if np.mean(delay) <= 1:
                    speedColor = "[#c4f129]"
                elif np.mean(delay) <= 10:
                    speedColor = "[#48a971]"
                elif np.mean(delay) <= 30:
                    speedColor = "[#494b9b]"
                else:
                    speedColor = "[#ab333d]"

                passed = "{:02d}:{:02d}".format(math.floor(sum(delay)/60), round(sum(delay))%60)
                remaining = "{:02d}:{:02d}".format(math.floor((total*np.mean(delay)-sum(delay))/60), round(total*np.mean(delay)-sum(delay))%60)

            speed = f" {iteration}/{total} at {measure} "
            prediction = f" {passed} < {remaining} "

            # Print single bar across two lines
            rprint(f'\r{f"{name}".center(prefix)} {"".join(line1)}{speedColor}{speed.center(suffix-1)}[white]')
            rprint(f'[#48a971]{f"{percent}%".center(prefix)}[/#48a971] {"".join(line2)}[#494b9b]{prediction.center(suffix-1)}', end = printEnd)
            delay.append(time.time())

            return delay

        # Print at 0 progress
        delay = []
        delay = printProgressBar(0, delay)
        down(int(position == "first")*2)
        # Update the progress bar
        for i, item in enumerate(iterable):
            yield item
            up(int(position == "first")*2+1)
            delay = printProgressBar(i + 1, delay)
            down(int(position == "first")*2)
            
        down(int(position != "first"))
    else:
        for i, item in enumerate(iterable):
            yield item

def load_model_from_config(model, verbose=False):
    # Load the model's state dictionary from the specified file
    try:
        # First try to load as a Safetensor, then as a pickletensor
        try:
            pl_sd = load_file(model, device="cpu")
        except: 
            rprint(f"[#ab333d]Model is not a Safetensor. Please consider using Safetensors format for better security.")
            pl_sd = torch.load(model, map_location="cpu")

        sd = pl_sd

        # If "state_dict" is found in the loaded dictionary, assign it to sd
        if 'state_dict' in sd:
            sd = pl_sd["state_dict"]

        return sd
    except Exception as e: 
                rprint(f"[#ab333d]{traceback.format_exc()}\n\nThis may indicate a model has not been downloaded fully, or is corrupted.")

def load_img(path, h0, w0):
    # Open the image at the specified path and prepare it for image to image
    image = Image.open(path).convert("RGB")
    w, h = image.size

    # Override the image size if h0 and w0 are provided
    if h0 is not None and w0 is not None:
        h, w = h0, w0

    # Adjust the width and height to be divisible by 8 and resize the image using bicubic resampling
    w, h = map(lambda x: x - x % 8, (w, h))
    image = image.resize((w, h), resample=Image.Resampling.BICUBIC)

    # Convert the image to a numpy array of float32 values in the range [0, 1], transpose it, and convert it to a PyTorch tensor
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)

    # Apply a normalization by scaling the values in the range [-1, 1]
    return 2.*image - 1.

def flatten(el):
    # Flatten nested elements by recursively traversing through children
    flattened = [flatten(children) for children in el.children()]
    res = [el]
    for c in flattened:
        res += c
    return res

def adjust_gamma(image, gamma=1.0):
    # Create a lookup table for the gamma function
    gamma_map = [255 * ((i / 255.0) ** (1.0 / gamma)) for i in range(256)]
    gamma_table = bytes([(int(x / 255.0 * 65535.0) >> 8) for x in gamma_map] * 3)

    # Apply the gamma correction using the lookup table
    return image.point(gamma_table)

def load_model(modelpath, modelfile, config, device, precision, optimized):
    timer = time.time()

    # Check the modelfile and print corresponding loading message
    print()
    if modelfile == "model.pxlm":
        print(f"Loading primary model")
    elif modelfile == "modelmicro.pxlm":
        print(f"Loading micro model")
    elif modelfile == "modelmini.pxlm":
        print(f"Loading mini model")
    elif modelfile == "modelmega.pxlm":
        print(f"Loading mega model")
    elif modelfile == "paletteGen.pxlm":
        print(f"Loading PaletteGen model")
    else:
        rprint(f"Loading custom model from [#48a971]{modelfile}")

    # Determine if turbo mode is enabled
    turbo = True
    if optimized == "true":
        turbo = False

    # Load the model's state dictionary from the specified file
    sd = load_model_from_config(f"{modelpath+modelfile}")

    # Separate the input and output blocks from the state dictionary
    li, lo = [], []
    for key, value in sd.items():
        sp = key.split(".")
        if (sp[0]) == "model":
            if "input_blocks" in sp:
                li.append(key)
            elif "middle_block" in sp:
                li.append(key)
            elif "time_embed" in sp:
                li.append(key)
            else:
                lo.append(key)

    # Reorganize the state dictionary keys to match the model structure
    for key in li:
        sd["model1." + key[6:]] = sd.pop(key)
    for key in lo:
        sd["model2." + key[6:]] = sd.pop(key)

    # Load the model configuration
    config = OmegaConf.load(f"{config}")

    global modelPV
    # Ignore an annoying userwaring
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Load the pixelvae
        modelPV = load_pixelvae_model(f"models/decoder/decoder.px", device, "eVWtlIBjTRr0-gyZB0smWSwxCiF8l4PVJcNJOIFLFqE=")

    # Instantiate and load the main model
    global model
    model = instantiate_from_config(config.modelUNet)
    _, _ = model.load_state_dict(sd, strict=False)
    model.eval()
    model.unet_bs = 1
    model.cdevice = device
    model.turbo = turbo
    tomesd.apply_patch(model, ratio=0.6, use_rand=True, merge_attn=True, merge_crossattn=True, merge_mlp=True)

    # Instantiate and load the conditional stage model
    global modelCS
    modelCS = instantiate_from_config(config.modelCondStage)
    _, _ = modelCS.load_state_dict(sd, strict=False)
    modelCS.eval()
    modelCS.cond_stage_model.device = device

    # Instantiate and load the first stage model
    global modelFS
    modelFS = instantiate_from_config(config.modelFirstStage)
    _, _ = modelFS.load_state_dict(sd, strict=False)
    modelFS.eval()

    # Set precision and device settings
    if device != "cpu" and precision == "autocast":
        model.half()
        modelCS.half()
        precision = "half"

    assign_lora_names_to_compvis_modules(model, modelCS)

    # Print loading information
    play("iteration.wav")
    rprint(f"[#c4f129]Loaded model to [#48a971]{model.cdevice}[#c4f129] at [#48a971]{precision} precision[#c4f129] in [#48a971]{round(time.time()-timer, 2)} [#c4f129]seconds")

def kCentroid(image, width, height, centroids):
    image = image.convert("RGB")

    # Create an empty array for the downscaled image
    downscaled = np.zeros((height, width, 3), dtype=np.uint8)

    # Calculate the scaling factors
    wFactor = image.width/width
    hFactor = image.height/height

    # Iterate over each tile in the downscaled image
    for x, y in product(range(width), range(height)):
            # Crop the tile from the original image
            tile = image.crop((x*wFactor, y*hFactor, (x*wFactor)+wFactor, (y*hFactor)+hFactor))

            # Quantize the colors of the tile using k-means clustering
            tile = tile.quantize(colors=centroids, method=1, kmeans=centroids).convert("RGB")

            # Get the color counts and find the most common color
            color_counts = tile.getcolors()
            most_common_color = max(color_counts, key=lambda x: x[0])[1]

            # Assign the most common color to the corresponding pixel in the downscaled image
            downscaled[y, x, :] = most_common_color

    return Image.fromarray(downscaled, mode='RGB')

def pixelDetect(image: Image):
    # Thanks to https://github.com/paultron for optimizing my garbage code 
    # I swapped the axis so they accurately reflect the horizontal and vertical scaling factor for images with uneven ratios

    # Convert the image to a NumPy array
    npim = np.array(image)[..., :3]

    # Compute horizontal differences between pixels
    hdiff = np.sqrt(np.sum((npim[:, :-1, :] - npim[:, 1:, :])**2, axis=2))
    hsum = np.sum(hdiff, 0)

    # Compute vertical differences between pixels
    vdiff = np.sqrt(np.sum((npim[:-1, :, :] - npim[1:, :, :])**2, axis=2))
    vsum = np.sum(vdiff, 1)

    # Find peaks in the horizontal and vertical sums
    hpeaks, _ = scipy.signal.find_peaks(hsum, distance=1, height=0.0)
    vpeaks, _ = scipy.signal.find_peaks(vsum, distance=1, height=0.0)
    
    # Compute spacing between the peaks
    hspacing = np.diff(hpeaks)
    vspacing = np.diff(vpeaks)

    # Resize input image using kCentroid with the calculated horizontal and vertical factors
    return kCentroid(image, round(image.width/np.median(hspacing)), round(image.height/np.median(vspacing)), 2)

def pixelDetectVerbose():
    # Check if input file exists and open it
    assert os.path.isfile("temp/input.png")
    init_img = Image.open("temp/input.png")

    rprint(f"\n[#48a971]Finding pixel ratio for current cel")

    # Process the image using pixelDetect and save the result
    for _ in clbar(range(1), name = "Processed", position = "last", unit = "image", prefixwidth = 12, suffixwidth = 28):
        downscale = pixelDetect(init_img)

        numColors = determine_best_k_verbose(downscale, 64)

        for _ in clbar([downscale], name = "Palettizing", position = "first", prefixwidth = 12, suffixwidth = 28): 
            img_indexed = downscale.quantize(colors=numColors, method=1, kmeans=numColors, dither=0).convert('RGB')
        
        img_indexed.save("temp/temp.png")
    play("batch.wav")

def kDenoise(image, smoothing, strength):
    image = image.convert("RGB")

    # Create an array to store the denoised image
    denoised = np.zeros((image.height, image.width, 3), dtype=np.uint8)

    # Iterate over each pixel
    for x, y in product(range(image.width), range(image.height)):
            # Crop the image to a 3x3 tile around the current pixel
            tile = image.crop((x-1, y-1, min(x+2, image.width), min(y+2, image.height)))

            # Calculate the number of centroids based on the tile size and strength
            centroids = max(2, min(round((tile.width*tile.height)*(1/strength)), (tile.width*tile.height)))

            # Quantize the tile to the specified number of centroids
            tile = tile.quantize(colors=centroids, method=1, kmeans=centroids).convert("RGB")

            # Get the color counts for each centroid and find the most common color
            color_counts = tile.getcolors()
            final_color = tile.getpixel((1, 1))

            # Check if the count of the most common color is below a threshold
            count = 0
            for ele in color_counts:
                if (ele[1] == final_color):
                    count = ele[0]

            # If the count is below the threshold, choose the most common color
            if count < 1+round(((tile.width*tile.height)*0.8)*(smoothing/10)):
                final_color = max(color_counts, key=lambda x: x[0])[1]
            
            # Store the final color in the downscaled image array
            denoised[y, x, :] = final_color

    return Image.fromarray(denoised, mode='RGB')

def determine_best_k(image, max_k):
    # Convert the image to RGB mode
    image = image.convert("RGB")

    # Prepare arrays for distortion calculation
    pixels = np.array(image)
    pixel_indices = np.reshape(pixels, (-1, 3))

    # Calculate distortion for different values of k
    distortions = []
    for k in range(1, max_k + 1):
        quantized_image = image.quantize(colors=k, method=2, kmeans=k, dither=0)
        centroids = np.array(quantized_image.getpalette()[:k * 3]).reshape(-1, 3)
        
        # Calculate distortions
        distances = np.linalg.norm(pixel_indices[:, np.newaxis] - centroids, axis=2)
        min_distances = np.min(distances, axis=1)
        distortions.append(np.sum(min_distances ** 2))

    # Calculate the rate of change of distortions
    rate_of_change = np.diff(distortions) / np.array(distortions[:-1])
    
    # Find the elbow point (best k value)
    if len(rate_of_change) == 0:
        best_k = 2
    else:
        elbow_index = np.argmax(rate_of_change)
        best_k = elbow_index + 2

    return best_k

def determine_best_palette_verbose(image, paletteFolder):
    # Convert the image to RGB mode
    image = image.convert("RGB")

    paletteImages = []
    paletteImages.extend(os.listdir(paletteFolder))

    # Prepare arrays for distortion calculation
    pixels = np.array(image)
    pixel_indices = np.reshape(pixels, (-1, 3))

    # Calculate distortion for different palettes
    distortions = []
    for palImg in clbar(paletteImages, name = "Searching", position = "first", prefixwidth = 12, suffixwidth = 28):
        try:
            palImg = Image.open(f"{paletteFolder}/{palImg}").convert('RGB')
        except:
            continue
        palette = []

        # Extract palette colors
        palColors = palImg.getcolors(16777216)
        numColors = len(palColors)
        palette = np.concatenate([x[1] for x in palColors]).tolist()
        
        # Create a new palette image
        palImg = Image.new('P', (256, 1))
        palImg.putpalette(palette)

        quantized_image = image.quantize(method=1, kmeans=numColors, palette=palImg, dither=0)
        centroids = np.array(quantized_image.getpalette()[:numColors * 3]).reshape(-1, 3)
        
        # Calculate distortions
        distances = np.linalg.norm(pixel_indices[:, np.newaxis] - centroids, axis=2)
        min_distances = np.min(distances, axis=1)
        distortions.append(np.sum(min_distances ** 2))
    
    # Find the best match
    best_match_index = np.argmin(distortions)
    best_palette = Image.open(f"{paletteFolder}/{paletteImages[best_match_index]}").convert('RGB')

    return best_palette, paletteImages[best_match_index]

def determine_best_k_verbose(image, max_k):
    # Convert the image to RGB mode
    image = image.convert("RGB")

    # Prepare arrays for distortion calculation
    pixels = np.array(image)
    pixel_indices = np.reshape(pixels, (-1, 3))

    # Calculate distortion for different values of k
    # Divided into 'chunks' for nice progress displaying
    distortions = []
    count = 0
    for k in clbar(range(4, round(max_k/8) + 2), name = "Finding K", position = "first", prefixwidth = 12, suffixwidth = 28):
        for n in range(round(max_k/k)):
            count += 1
            quantized_image = image.quantize(colors=count, method=2, kmeans=count, dither=0)
            centroids = np.array(quantized_image.getpalette()[:count * 3]).reshape(-1, 3)
            
            # Calculate distortions
            distances = np.linalg.norm(pixel_indices[:, np.newaxis] - centroids, axis=2)
            min_distances = np.min(distances, axis=1)
            distortions.append(np.sum(min_distances ** 2))

    # Calculate the rate of change of distortions
    rate_of_change = np.diff(distortions) / np.array(distortions[:-1])
    
    # Find the elbow point (best k value)
    if len(rate_of_change) == 0:
        best_k = 1
    else:
        elbow_index = np.argmax(rate_of_change)
        best_k = elbow_index + 2

    return best_k

def palettize(numFiles, source, colors, bestPaletteFolder, paletteFile, paletteURL, dithering, strength, denoise, smoothness, intensity):
    # Check if a palette URL is provided and try to download the palette image
    if source == "URL":
        try:
            paletteFile = BytesIO(requests.get(paletteURL).content)
            testImg = Image.open(paletteFile).convert('RGB')
        except:
            rprint(f"\n[#ab333d]ERROR: URL {paletteURL} cannot be reached or is not an image\nReverting to Adaptive palette")
            paletteFile = ""

    timer = time.time()

    # Create a list to store file paths
    files = []
    for n in range(numFiles):
        files.append(f"temp/input{n+1}.png")

    # Determine the number of colors based on the palette or user input
    if paletteFile != "":
        palImg = Image.open(paletteFile).convert('RGB')
        numColors = len(palImg.getcolors(16777216))
    else:
        numColors = colors

    # Create the string for conversion message
    string = f"\n[#48a971]Converting output[white] to [#48a971]{numColors}[white] colors"

    # Add dithering information if strength and dithering are greater than 0
    if strength > 0 and dithering > 0:
        string = f'{string} with order [#48a971]{dithering}[white] dithering'

    if source == "Automatic":
        string = f"\n[#48a971]Converting output[white] to best number of colors"
    elif source == "Best Palette":
        string = f"\n[#48a971]Converting output[white] to best color palette"

    # Print the conversion message
    rprint(string)

    palFiles = []
    # Process each file in the list
    for file in clbar(files, name = "Processed", position = "last", unit = "image", prefixwidth = 12, suffixwidth = 28):

        img = Image.open(file).convert('RGB')

        # Apply denoising if enabled
        if denoise == "true":
            img = kDenoise(img, smoothness, intensity)

        # Calculate the threshold for dithering
        threshold = 4*strength

        if source == "Automatic":
            numColors = determine_best_k_verbose(img, 64)
        
        # Check if a palette file is provided
        if (paletteFile != "" and os.path.isfile(file)) or source == "Best Palette":
            # Open the palette image and calculate the number of colors
            if source == "Best Palette":
                palImg, palFile = determine_best_palette_verbose(img, bestPaletteFolder)
                palFiles.append(str(palFile))
            else:
                palImg = Image.open(paletteFile).convert('RGB')
            
            numColors = len(palImg.getcolors(16777216))

            if strength > 0 and dithering > 0:
                for _ in clbar([img], name = "Palettizing", position = "first", prefixwidth = 12, suffixwidth = 28):
                    # Adjust the image gamma
                    img = adjust_gamma(img, 1.0-(0.02*strength))

                    # Extract palette colors
                    palette = [x[1] for x in palImg.getcolors(16777216)]

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        # Perform ordered dithering using Bayer matrix
                        palette = hitherdither.palette.Palette(palette)
                        img_indexed = hitherdither.ordered.bayer.bayer_dithering(img, palette, [threshold, threshold, threshold], order=dithering).convert('RGB')
            else:
                # Extract palette colors
                palette = np.concatenate([x[1] for x in palImg.getcolors(16777216)]).tolist()
                
                # Create a new palette image
                palImg = Image.new('P', (256, 1))
                palImg.putpalette(palette)

                # Perform quantization without dithering
                for _ in clbar([img], name = "Palettizing", position = "first", prefixwidth = 12, suffixwidth = 28):
                    img_indexed = img.quantize(method=1, kmeans=numColors, palette=palImg, dither=0).convert('RGB')

        elif numColors > 0 and os.path.isfile(file):
            if strength > 0 and dithering > 0:

                # Perform quantization with ordered dithering
                for _ in clbar([img], name = "Palettizing", position = "first", prefixwidth = 12, suffixwidth = 28):
                    img_indexed = img.quantize(colors=numColors, method=1, kmeans=numColors, dither=0).convert('RGB')

                    # Adjust the image gamma
                    img = adjust_gamma(img, 1.0-(0.03*strength))

                    # Extract palette colors
                    palette = [x[1] for x in img_indexed.getcolors(16777216)]

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        # Perform ordered dithering using Bayer matrix
                        palette = hitherdither.palette.Palette(palette)
                        img_indexed = hitherdither.ordered.bayer.bayer_dithering(img, palette, [threshold, threshold, threshold], order=dithering).convert('RGB')

            else:
                # Perform quantization without dithering
                for _ in clbar([img], name = "Palettizing", position = "first", prefixwidth = 12, suffixwidth = 28): 
                    img_indexed = img.quantize(colors=numColors, method=1, kmeans=numColors, dither=0).convert('RGB')

        img_indexed.save(file)

        if file != files[-1]:
            play("iteration.wav")
        else:
            play("batch.wav")

    rprint(f"[#c4f129]Palettized [#48a971]{len(files)}[#c4f129] images in [#48a971]{round(time.time()-timer, 2)}[#c4f129] seconds")
    if source == "Best Palette":
        rprint(f"[#c4f129]Palettes used: [#494b9b]{', '.join(palFiles)}")

def palettizeOutput(numFiles):
    # Create a list to store file paths
    files = []
    for n in range(numFiles):
        files.append(f"temp/temp{n+1}.png")
    
    # Process the image using pixelDetect and save the result
    for file in clbar(files, name = "Processed", position = "last", unit = "image", prefixwidth = 12, suffixwidth = 28):
        img = Image.open(file).convert('RGB')

        numColors = determine_best_k_verbose(img, 64)

        for _ in clbar([img], name = "Palettizing", position = "first", prefixwidth = 12, suffixwidth = 28): 
            img_indexed = img.quantize(colors=numColors, method=1, kmeans=numColors, dither=0).convert('RGB')
        
            img_indexed.save(file)
        if file != files[-1]:
            play("iteration.wav")
        else:
            play("batch.wav")

# Unused for now
def rembg(numFiles):
    
    timer = time.time()
    files = []

    rprint(f"\n[#48a971]Removing [#48a971]{numFiles}[white] backgrounds")
    
    # Create a list of file paths
    for n in range(numFiles):
        files.append(f"temp/input{n+1}.png")

    # Process each file in the list
    for file in clbar(files, name = "Processed", position = "", unit = "image", prefixwidth = 12, suffixwidth = 28):
        img = Image.open(file).convert('RGB')
        
        # Check if the file exists
        if os.path.isfile(file):
            # Ignore warnings during background removal
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Remove the background and save the image
                #remove(img).save(file)

            if file != files[-1]:
                play("iteration.wav")
            else:
                play("batch.wav")
    rprint(f"[#c4f129]Removed [#48a971]{len(files)}[#c4f129] backgrounds in [#48a971]{round(time.time()-timer, 2)}[#c4f129] seconds")

def kCentroidVerbose(width, height, centroids):
    # Check if the input file exists and open it
    assert os.path.isfile("temp/input.png")
    init_img = Image.open("temp/input.png")

    rprint(f"\n[#48a971]K-Centroid downscaling[white] from [#48a971]{init_img.width}[white]x[#48a971]{init_img.height}[white] to [#48a971]{width}[white]x[#48a971]{height}[white] with [#48a971]{centroids}[white] centroids")

    # Perform k-centroid downscaling and save the image
    for _ in clbar(range(1), name = "Processed", unit = "image", prefixwidth = 12, suffixwidth = 28):
        kCentroid(init_img, int(width), int(height), int(centroids)).save("temp/temp.png")
    play("batch.wav")
        
def paletteGen(colors, device, precision, prompt, seed):
    # Calculate the base for palette generation
    base = 2**round(math.log2(colors))

    # Calculate the width of the image based on the base and number of colors
    width = 512+((512/base)*(colors-base))

    # Generate text-to-image conversion with specified parameters
    txt2img(None, ["none"], [0], device, precision, 1, prompt, "", int(width), 512, 20, 7.0, int(seed), 1, "false", "false", "false", "false")

    # Open the generated image
    image = Image.open("temp/temp1.png").convert('RGB')

    # Perform k-centroid downscaling on the image
    image = kCentroid(image, int(image.width/(512/base)), 1, 2)

    # Iterate over the pixels in the image and set corresponding palette colors
    palette = Image.new('P', (colors, 1))
    for x in range(image.width):
        for y in range(image.height):
            r, g, b = image.getpixel((x, y))

            palette.putpixel((x, y), (r, g, b))

    palette.save("temp/temp1.png")
    rprint(f"[#c4f129]Image converted to color palette with [#48a971]{colors}[#c4f129] colors")

def txt2img(loraPath, loraFiles, loraWeights, device, precision, pixelSize, prompt, negative, W, H, ddim_steps, scale, seed, n_iter, tilingX, tilingY, pixelvae, post):
    os.makedirs("temp", exist_ok=True)
    outpath = "temp"

    timer = time.time()
    
    # Set the seed for random number generation if not provided
    if seed == None:
        seed = randint(0, 1000000)
    
    print()
    seed_everything(seed)
    rprint(f"[#48a971]Text to Image[white] generating for [#48a971]{n_iter}[white] iterations with [#48a971]{ddim_steps}[white] steps per iteration at [#48a971]{W}[white]x[#48a971]{H}")

    start_code = None
    sampler = "euler"

    assert prompt is not None
    data = [prompt]
    negative_data = [negative]

    global model
    global modelCS
    global modelFS
    global modelPV

    # Patch tiling for model and modelFS
    model, modelFS, modelPV = patch_tiling(tilingX, tilingY, model, modelFS, modelPV)

    # Set the precision scope based on device and precision
    if device != "cpu" and precision == "autocast":
        precision_scope = autocast
    else:
        precision_scope = nullcontext

    # !!! REMEMBER: ALL MODEL FILES ARE BOUND UNDER THE LICENSE AGREEMENTS OUTLINED HERE: https://astropulse.co/#retrodiffusioneula https://astropulse.co/#retrodiffusionmodeleula !!!
    loras = []
    fernet = Fernet("I47jl1hqUPug4KbVYd60_zeXhn_IH_ECT3QRGiBxdxo=")
    for i, loraFile in enumerate(loraFiles):
        if loraFile != "none":
            lora_filename = os.path.join(loraPath, loraFile)
            if os.path.splitext(loraFile)[1] == ".pxlm":
                with open(lora_filename, 'rb') as enc_file:
                    encrypted = enc_file.read()
                try:
                    decrypted = fernet.decrypt(encrypted)
                except:
                    decrypted = encrypted
                with open(lora_filename, 'wb') as dec_file:
                    dec_file.write(decrypted)
            loras.append(load_lora(lora_filename, model))
            loras[i].multiplier = loraWeights[i]/100
            register_lora_for_inference(loras[i])
            apply_lora()
            rprint(f"[#494b9b]Using [#48a971]{os.path.splitext(loraFile)[0]} [#494b9b]LoRA with [#48a971]{loraWeights[i]}% [#494b9b]strength")
        else:
            loras.append(None)


    seeds = []
    with torch.no_grad():
        base_count = 1
        # Iterate over the specified number of iterations
        for n in clbar(range(n_iter), name = "Iterations", position = "last", unit = "image", prefixwidth = 12, suffixwidth = 28):
            # Iterate over the prompts
            for prompts in data:
                # Use the specified precision scope
                with precision_scope("cuda"):
                    modelCS.to(device)
                    uc = None
                    uc = modelCS.get_learned_conditioning(negative_data)

                    c = modelCS.get_learned_conditioning(prompts)

                    shape = [1, 4, H // 8, W // 8]

                    # Move modelCS to CPU if necessary to free up GPU memory
                    if device != "cpu":
                        mem = torch.cuda.memory_allocated() / 1e6
                        modelCS.to("cpu")
                        # Wait until memory usage decreases
                        while torch.cuda.memory_allocated() / 1e6 >= mem:
                            time.sleep(1)

                    # Generate samples using the model
                    samples_ddim = model.sample(
                        S=ddim_steps,
                        conditioning=c,
                        seed=seed,
                        shape=shape,
                        verbose=False,
                        unconditional_guidance_scale=scale,
                        unconditional_conditioning=uc,
                        eta=0.0,
                        x_T=start_code,
                        sampler = sampler,
                    )

                    if pixelvae == "true":
                        # Pixel clustering mode, lower threshold means bigger clusters
                        denoise = 0.08
                        x_sample = modelPV.run_cluster(samples_ddim, threshold=denoise, wrap_x=bool(tilingX == "true"), wrap_y=bool(tilingY == "true"))
                        #x_sample = modelPV.run_plain(samples_ddim)
                        # Convert to numpy format, skip downscale later
                        x_sample = x_sample[0].cpu().numpy()
                    else:
                        modelFS.to(device)
                        # Decode the samples using the first stage of the model
                        x_sample = [modelFS.decode_first_stage(samples_ddim[i:i+1].to(device))[0].cpu() for i in range(samples_ddim.size(0))]
                        # Convert the list of decoded samples to a tensor and normalize the values to [0, 1]
                        x_sample = torch.stack(x_sample).float()
                        x_sample = torch.clamp((x_sample + 1.0) / 2.0, min=0.0, max=1.0)

                        # Rearrange the dimensions of the tensor and scale the values to the range [0, 255]
                        x_sample = 255.0 * rearrange(x_sample[0].cpu().numpy(), "c h w -> h w c")
                    
                    # Convert the numpy array to an image
                    x_sample_image = Image.fromarray(x_sample.astype(np.uint8))

                    file_name = "temp" + f"{base_count}"
                    if x_sample_image.width > int(W/pixelSize) and x_sample_image.height > int(H/pixelSize):
                        # Resize the image if pixel is true
                        x_sample_image = kCentroid(x_sample_image, int(W/pixelSize), int(H/pixelSize), 2)
                    elif x_sample_image.width < int(W/pixelSize) and x_sample_image.height < int(H/pixelSize):
                        x_sample_image = x_sample_image.resize((W, H), resample=Image.Resampling.NEAREST)

                    if "1bit.pxlm" in loraFiles:
                        x_sample_image = x_sample_image.quantize(colors=4, method=1, kmeans=4, dither=0).convert('RGB')
                        x_sample_image = x_sample_image.quantize(colors=2, method=1, kmeans=2, dither=0).convert('RGB')
                        pixels = list(x_sample_image.getdata())
                        darkest, brightest = min(pixels), max(pixels)
                        new_pixels = [0 if pixel == darkest else 255 if pixel == brightest else pixel for pixel in pixels]
                        new_image = Image.new("L", x_sample_image.size)
                        new_image.putdata(new_pixels)
                        x_sample_image = new_image.convert('RGB')

                    x_sample_image.save(
                        os.path.join(outpath, file_name + ".png")
                    )

                    if n_iter > 1 and base_count < n_iter:
                        play("iteration.wav")

                    seeds.append(str(seed))
                    seed += 1
                    base_count += 1

                    # Move modelFS to CPU if necessary to free up GPU memory
                    if device != "cpu" and modelFS.device != torch.device("cpu"):
                        mem = torch.cuda.memory_allocated() / 1e6
                        modelFS.to("cpu")
                        # Wait until memory usage decreases
                        while torch.cuda.memory_allocated() / 1e6 >= mem:
                            time.sleep(1)
                    
                    # Delete the samples to free up memory
                    del samples_ddim

        for i, lora in enumerate(loras):
            if lora is not None:
                # Release lora
                remove_lora_for_inference(lora)
            if os.path.splitext(loraFiles[i])[1] == ".pxlm":
                with open(os.path.join(loraPath, loraFiles[i]), 'rb') as enc_file:
                    decrypted = enc_file.read()
                encrypted = fernet.encrypt(decrypted)
                with open(os.path.join(loraPath, loraFiles[i]), 'wb') as dec_file:
                    dec_file.write(encrypted)
        del loras

        if post == "true":
            play("iteration.wav")
            palettizeOutput(int(n_iter))
        else:
            play("batch.wav")
        rprint(f"[#c4f129]Image generation completed in [#48a971]{round(time.time()-timer, 2)} [#c4f129]seconds\n[#48a971]Seeds: [#494b9b]{', '.join(seeds)}")

def img2img(loraPath, loraFiles, loraWeights, device, precision, pixelSize, prompt, negative, W, H, ddim_steps, scale, strength, seed, n_iter, tilingX, tilingY, pixelvae, post):
    timer = time.time()
    init_img = "temp/input.png"

    # Load initial image and move it to the specified device
    assert os.path.isfile(init_img)
    init_image = load_img(init_img, H, W).to(device)

    os.makedirs("temp", exist_ok=True)
    outpath = "temp"

    # Set a random seed if not provided
    if seed == None:
        seed = randint(0, 1000000)

    print()
    seed_everything(seed)
    rprint(f"[#48a971]Image to Image[white] generating for [#48a971]{n_iter}[white] iterations with [#48a971]{ddim_steps}[white] steps per iteration at [#48a971]{W}[white]x[#48a971]{H}")

    sampler = "ddim"

    assert prompt is not None
    data = [prompt]
    negative_data = [negative]

    global model
    global modelCS
    global modelFS
    global modelPV

    # Patch tiling for model and modelFS
    model, modelFS, modelPV = patch_tiling(tilingX, tilingY, model, modelFS, modelPV)

    # Move the modelFS to the specified device
    modelFS.to(device)

    # Repeat the initial image for batch processing
    init_image = repeat(init_image, "1 ... -> b ...", b=1)

    # Move the initial image to latent space and resize it
    init_latent = modelFS.get_first_stage_encoding(modelFS.encode_first_stage(init_image))
    init_latent = torch.nn.functional.interpolate(init_latent, size=(H // 8, W // 8), mode="bilinear")

    # Move modelFS to CPU if necessary to free up GPU memory
    if device != "cpu":
        mem = torch.cuda.memory_allocated(device=device) / 1e6
        modelFS.to("cpu")
        # Wait until memory usage decreases
        while torch.cuda.memory_allocated(device=device) / 1e6 >= mem:
            time.sleep(1)

    # Set the precision scope based on device and precision
    if device != "cpu" and precision == "autocast":
        precision_scope = autocast
    else:
        precision_scope = nullcontext

    # !!! REMEMBER: ALL MODEL FILES ARE BOUND UNDER THE LICENSE AGREEMENTS OUTLINED HERE: https://astropulse.co/#retrodiffusioneula https://astropulse.co/#retrodiffusionmodeleula !!!
    loras = []
    fernet = Fernet("I47jl1hqUPug4KbVYd60_zeXhn_IH_ECT3QRGiBxdxo=")
    for i, loraFile in enumerate(loraFiles):
        if loraFile != "none":
            lora_filename = os.path.join(loraPath, loraFile)
            if os.path.splitext(loraFile)[1] == ".pxlm":
                with open(lora_filename, 'rb') as enc_file:
                    encrypted = enc_file.read()
                try:
                    decrypted = fernet.decrypt(encrypted)
                except:
                    decrypted = encrypted
                with open(lora_filename, 'wb') as dec_file:
                    dec_file.write(decrypted)
            loras.append(load_lora(lora_filename, model))
            loras[i].multiplier = loraWeights[i]/100
            register_lora_for_inference(loras[i])
            apply_lora()
            rprint(f"[#494b9b]Using [#48a971]{os.path.splitext(loraFile)[0]} [#494b9b]LoRA with [#48a971]{loraWeights[i]}% [#494b9b]strength")
        else:
            loras.append(None)

    seeds = []
    assert 0.0 <= strength <= 1.0, "can only work with strength in [0.0, 1.0]"

    # Calculate the number of steps for encoding
    t_enc = int(strength * ddim_steps)

    with torch.no_grad():
        base_count = 1

        # Iterate over the specified number of iterations
        for n in clbar(range(n_iter), name = "Iterations", position = "last", unit = "image", prefixwidth = 12, suffixwidth = 28):
            # Iterate over the prompts
            for prompts in data:
                # Use the specified precision scope
                with precision_scope("cuda"):
                    modelCS.to(device)
                    uc = None
                    uc = modelCS.get_learned_conditioning(negative_data)

                    c = modelCS.get_learned_conditioning(prompts)

                    # Move modelCS to CPU if necessary to free up GPU memory
                    if device != "cpu":
                        mem = torch.cuda.memory_allocated(device=device) / 1e6
                        modelCS.to("cpu")
                        # Wait until memory usage decreases
                        while torch.cuda.memory_allocated(device=device) / 1e6 >= mem:
                            time.sleep(1)

                    # Encode the scaled latent
                    z_enc = model.stochastic_encode(
                        init_latent,
                        torch.tensor([t_enc]).to(device),
                        seed,
                        0.0,
                        ddim_steps,
                    )
                    
                    # Generate samples using the model
                    samples_ddim = model.sample(
                        t_enc,
                        c,
                        z_enc,
                        unconditional_guidance_scale=scale,
                        unconditional_conditioning=uc,
                        sampler = sampler
                    )

                    if pixelvae == "true":
                        # Pixel clustering mode, lower threshold means bigger clusters
                        denoise = 0.08
                        x_sample = modelPV.run_cluster(samples_ddim, threshold=denoise, wrap_x=bool(tilingX == "true"), wrap_y=bool(tilingY == "true"))
                        #x_sample = modelPV.run_plain(samples_ddim)
                        # Convert to numpy format, skip downscale later
                        x_sample = x_sample[0].cpu().numpy()
                    else:
                        modelFS.to(device)
                        # Decode the samples using the first stage of the model
                        x_sample = [modelFS.decode_first_stage(samples_ddim[i:i+1].to(device))[0].cpu() for i in range(samples_ddim.size(0))]
                        # Convert the list of decoded samples to a tensor and normalize the values to [0, 1]
                        x_sample = torch.stack(x_sample).float()
                        x_sample = torch.clamp((x_sample + 1.0) / 2.0, min=0.0, max=1.0)

                        # Rearrange the dimensions of the tensor and scale the values to the range [0, 255]
                        x_sample = 255.0 * rearrange(x_sample[0].cpu().numpy(), "c h w -> h w c")
                    
                    # Convert the numpy array to an image
                    x_sample_image = Image.fromarray(x_sample.astype(np.uint8))

                    file_name = "temp" + f"{base_count}"
                    if x_sample_image.width > int(W/pixelSize) and x_sample_image.height > int(H/pixelSize):
                        # Resize the image if pixel is true
                        x_sample_image = kCentroid(x_sample_image, int(W/pixelSize), int(H/pixelSize), 2)
                    elif x_sample_image.width < int(W/pixelSize) and x_sample_image.height < int(H/pixelSize):
                        x_sample_image = x_sample_image.resize((W, H), resample=Image.Resampling.NEAREST)

                    if "1bit.pxlm" in loraFiles:
                        x_sample_image = x_sample_image.quantize(colors=4, method=1, kmeans=4, dither=0).convert('RGB')
                        x_sample_image = x_sample_image.quantize(colors=2, method=1, kmeans=2, dither=0).convert('RGB')
                        pixels = list(x_sample_image.getdata())
                        darkest, brightest = min(pixels), max(pixels)
                        new_pixels = [0 if pixel == darkest else 255 if pixel == brightest else pixel for pixel in pixels]
                        new_image = Image.new("L", x_sample_image.size)
                        new_image.putdata(new_pixels)
                        x_sample_image = new_image.convert('RGB')

                    x_sample_image.save(
                        os.path.join(outpath, file_name + ".png")
                    )

                    if n_iter > 1 and base_count < n_iter:
                        play("iteration.wav")

                    seeds.append(str(seed))
                    seed += 1
                    base_count += 1

                    # Move modelFS to CPU if necessary to free up GPU memory
                    if device != "cpu" and modelFS.device != torch.device("cpu"):
                        mem = torch.cuda.memory_allocated() / 1e6
                        modelFS.to("cpu")
                        # Wait until memory usage decreases
                        while torch.cuda.memory_allocated() / 1e6 >= mem:
                            time.sleep(1)
                    
                    # Delete the samples to free up memory
                    del samples_ddim

        for i, lora in enumerate(loras):
            if lora is not None:
                # Release lora
                remove_lora_for_inference(lora)
            if os.path.splitext(loraFiles[i])[1] == ".pxlm":
                with open(os.path.join(loraPath, loraFiles[i]), 'rb') as enc_file:
                    decrypted = enc_file.read()
                encrypted = fernet.encrypt(decrypted)
                with open(os.path.join(loraPath, loraFiles[i]), 'wb') as dec_file:
                    dec_file.write(encrypted)
        del loras

        if post == "true":
            play("iteration.wav")
            palettizeOutput(int(n_iter))
        else:
            play("batch.wav")
        rprint(f"[#c4f129]Image generation completed in [#48a971]{round(time.time()-timer, 2)} seconds\n[#48a971]Seeds: [#494b9b]{', '.join(seeds)}")

async def server(websocket):
    background = False

    async for message in websocket:
        if re.search(r"txt2img.+", message):
            await websocket.send("running txt2img")
            try:
                # Extract parameters from the message
                loraPath, loraFiles, loraWeights, device, precision, pixelSize, prompt, negative, w, h, ddim_steps, scale, seed, n_iter, tilingX, tilingY, pixelvae, post = searchString(message, "dlorapath", "dlorafiles", "dloraweights", "ddevice", "dprecision", "dpixelsize", "dprompt", "dnegative", "dwidth", "dheight", "dstep", "dscale", "dseed", "diter", "dtilingx", "dtilingy", "dpixelvae", "dpalettize", "end")
                loraFiles = loraFiles.split('|')
                loraWeights = [int(x) for x in loraWeights.split('|')]
                txt2img(loraPath, loraFiles, loraWeights, device, precision, int(pixelSize), prompt, negative, int(w), int(h), int(ddim_steps), float(scale), int(seed), int(n_iter), tilingX, tilingY, pixelvae, post)
                await websocket.send("returning txt2img")
            except Exception as e: 
                rprint(f"\n[#ab333d]ERROR:\n{traceback.format_exc()}")
                play("error.wav")
                await websocket.send("returning error")

        elif re.search(r"txt2pal.+", message):
            await websocket.send("running txt2pal")
            try:
                # Extract parameters from the message
                device, precision, prompt, seed, colors = searchString(message, "ddevice", "dprecision", "dprompt", "dseed", "dcolors", "end")
                paletteGen(int(colors), device, precision, prompt, int(seed))
                await websocket.send("returning txt2pal")
            except Exception as e:
                rprint(f"\n[#ab333d]ERROR:\n{traceback.format_exc()}")
                play("error.wav")
                await websocket.send("returning error")

        elif re.search(r"img2img.+", message):
            await websocket.send("running img2img")
            try:
                # Extract parameters from the message
                loraPath, loraFiles, loraWeights, device, precision, pixelSize, prompt, negative, w, h, ddim_steps, scale, strength, seed, n_iter, tilingX, tilingY, pixelvae, post = searchString(message, "dlorapath", "dlorafiles", "dloraweights", "ddevice", "dprecision", "dpixelsize", "dprompt", "dnegative", "dwidth", "dheight", "dstep", "dscale", "dstrength", "dseed", "diter", "dtilingx", "dtilingy", "dpixelvae", "dpalettize", "end")
                loraFiles = loraFiles.split('|')
                loraWeights = [int(x) for x in loraWeights.split('|')]
                img2img(loraPath, loraFiles, loraWeights, device, precision, int(pixelSize), prompt, negative, int(w), int(h), int(ddim_steps), float(scale), float(strength)/100, int(seed), int(n_iter), tilingX, tilingY, pixelvae, post)
                await websocket.send("returning img2img")
            except Exception as e: 
                rprint(f"\n[#ab333d]ERROR:\n{traceback.format_exc()}")
                play("error.wav")
                await websocket.send("returning error")

        elif re.search(r"palettize.+", message):
            await websocket.send("running palettize")
            try:
                # Extract parameters from the message
                numFiles, source, colors, bestPaletteFolder, paletteFile, paletteURL, dithering, strength, denoise, smoothness, intensity = searchString(message, "dnumfiles", "dsource", "dcolors", "dbestpalettefolder", "dpalettefile", "dpaletteURL", "ddithering", "dstrength", "ddenoise", "dsmoothness", "dintensity", "end")
                palettize(int(numFiles), source,  int(colors), bestPaletteFolder, paletteFile, paletteURL, int(dithering), int(strength), denoise, int(smoothness), int(intensity))
                await websocket.send("returning palettize")
            except Exception as e: 
                rprint(f"\n[#ab333d]ERROR:\n{traceback.format_exc()}")
                play("error.wav")
                await websocket.send("returning error")

        elif re.search(r"rembg.+", message):
            await websocket.send("running rembg")
            try:
                # Extract parameters from the message
                numFiles = searchString(message, "dnumfiles", "end")
                rembg(int(numFiles[0]))
                await websocket.send("returning rembg")
            except Exception as e: 
                rprint(f"\n[#ab333d]ERROR:\n{traceback.format_exc()}")
                play("error.wav")
                await websocket.send("returning error")

        elif re.search(r"pixelDetect.+", message):
            await websocket.send("running pixelDetect")
            try:
                pixelDetectVerbose()
                await websocket.send("returning pixelDetect")
            except Exception as e: 
                rprint(f"\n[#ab333d]ERROR:\n{traceback.format_exc()}")
                play("error.wav")
                await websocket.send("returning error")

        elif re.search(r"kcentroid.+", message):
            await websocket.send("running kcentroid")
            try:
                # Extract parameters from the message
                width, height, centroids = searchString(message, "dwidth", "dheight", "dcentroids", "end")
                kCentroidVerbose(int(width), int(height), int(centroids))
                await websocket.send("returning kcentroid")
            except Exception as e: 
                rprint(f"\n[#ab333d]ERROR:\n{traceback.format_exc()}")
                play("error.wav")
                await websocket.send("returning error")

        elif re.search(r"load.+", message):
            await websocket.send("loading model")
            global loaded
            if loaded != message:
                try:
                    # Extract parameters from the message
                    device, optimized, precision, path, model = searchString(message, "ddevice", "doptimized", "dprecision", "dpath", "dmodel", "end")
                    load_model(path, model, "scripts/v1-inference.yaml", device, precision, optimized)
                    loaded = message
                except Exception as e:
                    rprint(f"\n[#ab333d]ERROR:\n{traceback.format_exc()}")
                    play("error.wav")
                    await websocket.send("returning error")
                
            await websocket.send("loaded model")

        elif re.search(r"connected.+", message):
            global sounds
            try:
                background, sounds, extensionVersion = searchString(message, "dbackground", "dsound", "dversion", "end")
                rd = gw.getWindowsWithTitle("Retro Diffusion Image Generator")[0]
                if background == "false":
                    try:
                        # Restore and activate the window
                        rd.restore()
                        rd.activate()
                    except:
                        pass
                else:
                    try:
                        # Minimize the window
                        rd.minimize()
                    except:
                        pass
            except:
                pass

            if extensionVersion == expectedVersion:
                play("click.wav")
                await websocket.send("connected")
            else:
                rprint(f"\n[#ab333d]The current client is on a version that is incompatible with the image generator version. Please update the extension.")
                
        elif message == "no model":
            await websocket.send("loaded model")
        elif message == "recieved":
            if background == "false":
                try:
                    rd = gw.getWindowsWithTitle("Retro Diffusion Image Generator")[0]
                    if gw.getActiveWindow() is not None:
                        if gw.getActiveWindow().title == "Retro Diffusion Image Generator":
                            # Minimize the window
                            rd.minimize()
                except:
                    pass
            await websocket.send("free")
            torch.cuda.empty_cache()
        elif message == "shutdown":
            rprint("[#ab333d]Shutting down...")
            global running
            global timeout
            running = False
            await websocket.close()
            asyncio.get_event_loop().call_soon_threadsafe(asyncio.get_event_loop().stop)

async def connectSend(uri, message):
    async with connect(uri) as websocket:
        # Send a message over the WebSocket connection
        await websocket.send(message)

if system == "Windows":
    os.system("title Retro Diffusion Image Generator")
elif system == "Darwin":
    os.system("printf '\\033]0;Retro Diffusion Image Generator\\007'")
else:
    os.system("echo '\\033]0;Retro Diffusion Image Generator\\007'")

try:
    subprocess.run(['git', 'switch', '-f', expectedVersion], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    subprocess.run(['git', 'checkout', '.'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print("Updated local files")
except:
    print("Local files could not be updated, this is safe to ignore")

rprint("\n" + climage(Image.open("logo.png"), "centered") + "\n\n")

rprint("[#48a971]Starting Image Generator...")

start_server = serve(server, "localhost", 8765)

rprint("[#c4f129]Connected")

timeout = 1

# Run the server until it is completed
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()
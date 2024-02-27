print("Importing libraries. This may take one or more minutes.")

try:
    # Import core libraries
    import os, re, time, sys, asyncio, ctypes, math, threading, platform, json, sys, contextlib
    import torch
    import scipy
    import numpy as np
    from random import randint
    from omegaconf import OmegaConf
    from PIL import Image, ImageEnhance, ImageFilter
    import cv2
    from itertools import product
    from einops import rearrange
    from pytorch_lightning import seed_everything
    from transformers import BlipProcessor, BlipForConditionalGeneration, set_seed
    from typing import Optional
    from safetensors.torch import load_file
    from cryptography.fernet import Fernet

    # Import built libraries
    from ldm.util import instantiate_from_config, max_tile
    from optimization.pixelvae import load_pixelvae_model
    from optimization.taesd import TAESD
    from lora import (
        apply_lora,
        assign_lora_names_to_compvis_modules,
        load_lora,
        load_lora_raw,
        register_lora_for_inference,
        remove_lora_for_inference,
    )
    from upsample_prompts import load_chat_pipeline, upsample_caption, collect_response
    import segmenter
    import hitherdither

    # Import PyTorch functions
    from torch import autocast
    from torch import Tensor
    from torch.nn import functional as F
    from torch.nn.modules.utils import _pair

    # Import logging libraries
    import traceback, warnings
    import logging as pylog
    from transformers.utils import logging

    # Import websocket tools
    import requests
    from websockets import serve
    from io import BytesIO
    import base64

    # Import CLDM requirements
    from cldm_inference import load_controlnet, sample_cldm, unload_cldm

    # Import console management libraries
    from rich import print as rprint

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            import pygetwindow as gw
        except:
            pass
    from colorama import just_fix_windows_console
    import playsound

    system = platform.system()

    if system == "Windows":
        # Fix windows console for color codes
        just_fix_windows_console()

        # Patch existing console to remove interactivity
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-10), 128)

    # Disable all logging for pytorch lightning
    log = pylog.getLogger("lightning_fabric")
    log.propagate = False
    log.setLevel(pylog.ERROR)
    logging.set_verbosity(logging.CRITICAL)

except:
    import traceback

    print(f"ERROR:\n{traceback.format_exc()}")
    input(
        "Catastrophic failure, send this error to the developer.\nPress any key to exit."
    )
    exit()

# Global variables
global modelName
modelName = None
global modelSettings
modelSettings = None
# Unet
global model
# Conditioning (clip)
global modelCS
# VAE (Unused, replaced by TAE)
global modelFS
# TAE
global modelTA
# Pixel VAE
global modelPV
# Language model
global modelLM
modelLM = None
# Image classifier
global modelBLIP
modelBLIP = None
global modelType
global running
global loadedDevice
loadedDevice = "cpu"
global modelPath

global system_models
system_models = ["quality", "resfix", "crop", "detail", "brightness", "contrast", "saturation", "outline", "color_cr", "color_mg", "color_yb", "light_bf", "light_du", "light_lr"]

global sounds
sounds = False

expectedVersion = "10.5.0"

global maxSize

# model loading globals
global split_loaded
split_loaded = False

# For testing only, limits memory usage to "maxMemory"
maxSize = 512
maxMemory = 4
if False:
    cardMemory = torch.cuda.get_device_properties("cuda").total_memory / 1073741824
    usedMemory = cardMemory - (torch.cuda.mem_get_info()[0] / 1073741824)

    fractionalMaxMemory = (maxMemory - (usedMemory + 0.3)) / cardMemory
    print(usedMemory)
    print(cardMemory)
    print(maxMemory)
    print(cardMemory * fractionalMaxMemory)

    torch.cuda.set_per_process_memory_fraction(fractionalMaxMemory)

global timeout
global loaded
loaded = ""


# Clears pytorch and mps cache
def clearCache():
    global loadedDevice
    torch.cuda.empty_cache()
    if torch.backends.mps.is_available() and loadedDevice != "cpu":
        try:
            torch.mps.empty_cache()
        except:
            pass


# Play sound file
def audioThread(file):
    try:
        absoluteFile = os.path.abspath(f"../sounds/{file}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            playsound.playsound(absoluteFile)
    except:
        pass


# Async audio manager
def play(file):
    global sounds
    if sounds:
        try:
            threading.Thread(target=audioThread, args=(file,), daemon=True).start()
        except:
            pass


# Calculate precision mode by gpu
def get_precision(device, precision):
    fp16_mode = torch.bfloat16
    fp8_mode = torch.float8_e4m3fn
    if device == "cuda" and torch.cuda.is_available():
        # If GPU is nvidia 10xx force fp32 precision
        gpu_name = torch.cuda.get_device_name(device)
        if gpu_name.startswith("NVIDIA GeForce GTX 10"):
            if device == "cuda" and (precision == "fp8" or precision == "fp16"):
                precision = "fp32"

        # If GPU is nvidia 16xx, use float16 and enable benchmark mode
        elif torch.cuda.get_device_capability(device) == (7, 5) and gpu_name.startswith("NVIDIA GeForce GTX 16"):
            torch.backends.cudnn.benchmark = True
            fp16_mode = torch.float16
            if device == "cuda" and (precision == "fp8" or precision == "fp16"):
                precision = "fp16"

        # If GPU is nvidia 20xx disable float8 precision
        elif gpu_name.startswith("NVIDIA GeForce GTX 20"):
            if device == "cuda" and (precision == "fp8" or precision == "fp16"):
                fp16_mode = torch.float16
                precision = "fp16"
        
        # If GPU is not nvidia
        elif not "NVIDIA" in gpu_name:
            fp16_mode = torch.float16
            precision = "fp32"
    else:
        # Fallback to fp32 precision
        fp16_mode = torch.float16
        precision = "fp32"

    return precision, fp16_mode, fp8_mode


# Determine correct autocast mode
def autocast(device, precision, dtype = torch.float16):
    if device == "cuda" and torch.cuda.is_available():
        gpu_properties = torch.cuda.get_device_properties(device)
        gpu_name = gpu_properties.name
        if "NVIDIA" in gpu_name:
            if re.search(r"1[06]\d{2}", gpu_name):
                # Get manual autocast working
                return contextlib.nullcontext()
            else:
                if precision == "fp32":
                    return contextlib.nullcontext()
                else:
                    return torch.autocast("cuda", dtype=dtype, enabled=True)
        else:
            # Get manual autocast working
            return contextlib.nullcontext()
            
    if device == "cpu" or device == "mps" or precision == "fp32":
        return contextlib.nullcontext()
    
    return contextlib.nullcontext()


# Patch the Conv2d class with a custom __init__ method
def patch_conv(**patch):
    cls = torch.nn.Conv2d
    init = cls.__init__

    def __init__(self, *args, **kwargs):
        # Call the original init method and apply the patch arguments
        return init(self, *args, **kwargs, **patch)

    cls.__init__ = __init__


# Patch Conv2d layers in the given model for asymmetric padding
def patch_conv_asymmetric(model, x, y):
    for layer in flatten(model):
        if type(layer) == torch.nn.Conv2d:
            # Set padding mode based on x and y arguments
            layer.padding_modeX = "circular" if x else "constant"
            layer.padding_modeY = "circular" if y else "constant"

            # Compute padding values based on reversed padding repeated twice
            layer.paddingX = (
                layer._reversed_padding_repeated_twice[0],
                layer._reversed_padding_repeated_twice[1],
                0,
                0,
            )
            layer.paddingY = (
                0,
                0,
                layer._reversed_padding_repeated_twice[2],
                layer._reversed_padding_repeated_twice[3],
            )

            # Patch the _conv_forward method with a replacement function
            layer._conv_forward = __replacementConv2DConvForward.__get__(
                layer, torch.nn.Conv2d
            )


# Restore original _conv_forward method for Conv2d layers in the model
def restoreConv2DMethods(model):
    for layer in flatten(model):
        if type(layer) == torch.nn.Conv2d:
            layer._conv_forward = torch.nn.Conv2d._conv_forward.__get__(
                layer, torch.nn.Conv2d
            )


# Replacement function for Conv2d's _conv_forward method
def __replacementConv2DConvForward(
    self, input: Tensor, weight: Tensor, bias: Optional[Tensor]
):
    working = F.pad(input, self.paddingX, mode=self.padding_modeX)
    working = F.pad(working, self.paddingY, mode=self.padding_modeY)
    return F.conv2d(
        working, weight, bias, self.stride, _pair(0), self.dilation, self.groups
    )


# Patch Conv2d layers in the given models for asymmetric padding
def patch_tiling(tilingX, tilingY, model, modelTA, modelPV):
    # Patch for relevant models
    patch_conv_asymmetric(model, tilingX, tilingY)
    patch_conv_asymmetric(modelTA, tilingX, tilingY)
    patch_conv_asymmetric(modelPV.model, tilingX, tilingY)

    if tilingX or tilingY:
        # Print a message indicating the direction(s) patched for tiling
        rprint(
            "[#494b9b]Patched for tiling in the [#48a971]"
            + "X" * tilingX
            + "[#494b9b] and [#48a971]" * (tilingX and tilingY)
            + "Y" * tilingY
            + "[#494b9b] direction"
            + "s" * (tilingX and tilingY)
        )

    return model, modelTA, modelPV


def remove_repeated_words(string):
    # Splitting the string by spaces to preserve original punctuation
    parts = string.split()
    normalized_parts = []
    separators = []
    
    # Normalize parts and remember original separators
    for part in parts:
        if part.endswith(","):
            normalized_parts.append(part[:-1])
            separators.append(", ")
        else:
            normalized_parts.append(part)
            separators.append(" ")
    
    # Check for repetitions from the end
    if len(normalized_parts) > 1:
        i = -2
        while -i <= len(normalized_parts) and normalized_parts[-1] == normalized_parts[i]:
            i -= 1
        if i != -2:
            # Keep one instance of the repeated word
            final_parts = normalized_parts[:i+2]
        else:
            final_parts = normalized_parts
    else:
        final_parts = normalized_parts
    
    # Reconstruct the string using the original separators
    reconstructed_string = ""
    for i, part in enumerate(final_parts):
        if i < len(separators) - 1:  # Avoid index out of range
            reconstructed_string += part + separators[i]
        else:
            reconstructed_string += part  # Last part, no separator
    
    # Handling trailing separators if the last part was a repetition
    if reconstructed_string.endswith(", "):
        reconstructed_string = reconstructed_string[:-2]
    elif reconstructed_string.endswith(" "):
        reconstructed_string = reconstructed_string[:-1]
    
    return reconstructed_string


# Print image in console
def climage(image, alignment, *args):
    # Get console bounds with a small margin - better safe than sorry
    twidth, theight = (
        os.get_terminal_size().columns - 1,
        (os.get_terminal_size().lines - 1) * 2,
    )

    # Set up variables
    image = image.convert("RGBA")
    iwidth, iheight = min(twidth, image.width), min(theight, image.height)
    line = []
    lines = []

    # Alignment stuff

    margin = 0
    if alignment == "centered":
        margin = int((twidth / 2) - (iwidth / 2))
    elif alignment == "right":
        margin = int(twidth - iwidth)
    elif alignment == "manual":
        margin = args[0]

    # Loop over the height of the image / 2 (because 2 pixels = 1 text character)
    for y2 in range(int(iheight / 2)):
        # Add default colors to the start of the line
        line = [" " * margin]

        # Loop over width
        for x in range(iwidth):
            # Get the color for the upper and lower half of the text character
            r, g, b, a = image.getpixel((x, (y2 * 2)))
            r2, g2, b2, a2 = image.getpixel((x, (y2 * 2) + 1))

            # Set text characters, nothing, full block, half block. Half block + background color = 2 pixels
            if a < 200 and a2 < 200:
                line.append(f" ")
            else:
                # Convert to hex colors for Rich to use
                rgb, rgb2 = "#{:02x}{:02x}{:02x}".format(
                    r, g, b
                ), "#{:02x}{:02x}{:02x}".format(r2, g2, b2)

                # Lookup table because I was bored
                colorCodes = [
                    f"{rgb2} on {rgb}",
                    f"{rgb2}",
                    f"{rgb}",
                    "nothing",
                    f"{rgb}",
                ]
                # ~It just works~
                maping = (
                    int(a < 200)
                    + (int(a2 < 200) * 2)
                    + (int(rgb == rgb2 and a + a2 > 400) * 4)
                )
                color = colorCodes[maping]

                if rgb == rgb2:
                    line.append(f"[{color}]█[/]")
                else:
                    if maping == 2:
                        line.append(f"[{color}]▀[/]")
                    else:
                        line.append(f"[{color}]▄[/]")

        # Add default colors to the end of the line
        lines.append("".join(line) + "\u202F")
    return " \n".join(lines)


# Print progress bar in console
def clbar(
    iterable,
    name="",
    printEnd="\r",
    position="",
    unit="it",
    disable=False,
    prefixwidth=1,
    suffixwidth=1,
    total=0,
):
    # Console manipulation stuff
    def up(lines=1):
        for _ in range(lines):
            sys.stdout.write("\x1b[1A")
            sys.stdout.flush()

    def down(lines=1):
        for _ in range(lines):
            sys.stdout.write("\n")
            sys.stdout.flush()

    # Allow the complete disabling of the progress bar
    if not disable:
        # Positions the bar correctly
        down(int(position == "last") * 2)
        up(int(position == "first") * 3)

        # Set up variables
        if total > 0:
            # iterable = iterable[0:total]
            pass
        else:
            total = max(1, len(iterable))
        name = f"{name}"
        speed = f" {total}/{total} at 100.00 {unit}/s "
        prediction = f" 00:00 < 00:00 "
        prefix = max(len(name), len("100%"), prefixwidth)
        suffix = max(len(speed), len(prediction), suffixwidth)
        barwidth = os.get_terminal_size().columns - (suffix + prefix + 2)

        # Prints the progress bar
        def printProgressBar(iteration, delay):
            # Define progress bar graphic
            line1 = [
                "[#494b9b on #3b1725]▄[/#494b9b on #3b1725]",
                "[#c4f129 on #494b9b]▄[/#c4f129 on #494b9b]"
                * int(int(barwidth * min(total, iteration) // total) > 0),
                "[#ffffff on #494b9b]▄[/#ffffff on #494b9b]"
                * max(0, int(barwidth * min(total, iteration) // total) - 2),
                "[#c4f129 on #494b9b]▄[/#c4f129 on #494b9b]"
                * int(int(barwidth * min(total, iteration) // total) > 1),
                "[#3b1725 on #494b9b]▄[/#3b1725 on #494b9b]"
                * max(0, barwidth - int(barwidth * min(total, iteration) // total)),
                "[#494b9b on #3b1725]▄[/#494b9b on #3b1725]",
            ]
            line2 = [
                "[#3b1725 on #494b9b]▄[/#3b1725 on #494b9b]",
                "[#494b9b on #48a971]▄[/#494b9b on #48a971]"
                * int(int(barwidth * min(total, iteration) // total) > 0),
                "[#494b9b on #c4f129]▄[/#494b9b on #c4f129]"
                * max(0, int(barwidth * min(total, iteration) // total) - 2),
                "[#494b9b on #48a971]▄[/#494b9b on #48a971]"
                * int(int(barwidth * min(total, iteration) // total) > 1),
                "[#494b9b on #3b1725]▄[/#494b9b on #3b1725]"
                * max(0, barwidth - int(barwidth * min(total, iteration) // total)),
                "[#3b1725 on #494b9b]▄[/#3b1725 on #494b9b]",
            ]

            percent = ("{0:.0f}").format(100 * (min(total, iteration) / float(total)))

            # Avoid predicting speed until there's enough data
            if len(delay) >= 1:
                delay.append(time.time() - delay[-1])
                del delay[-2]

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

                passed = "{:02d}:{:02d}".format(
                    math.floor(sum(delay) / 60), round(sum(delay)) % 60
                )
                remaining = "{:02d}:{:02d}".format(
                    math.floor((total * np.mean(delay) - sum(delay)) / 60),
                    round(total * np.mean(delay) - sum(delay)) % 60,
                )

            speed = f" {min(total, iteration)}/{total} at {measure} "
            prediction = f" {passed} < {remaining} "

            # Print single bar across two lines
            rprint(
                f'\r{f"{name}".center(prefix)} {"".join(line1)}{speedColor}{speed.center(suffix-1)}[white]'
            )
            rprint(
                f'[#48a971]{f"{percent}%".center(prefix)}[/#48a971] {"".join(line2)}[#494b9b]{prediction.center(suffix-1)}',
                end=printEnd,
            )
            delay.append(time.time())

            return delay

        # Print at 0 progress
        delay = []
        delay = printProgressBar(0, delay)
        down(int(position == "first") * 2)
        # Update the progress bar
        for i, item in enumerate(iterable):
            yield item
            up(int(position == "first") * 2 + 1)
            delay = printProgressBar(i + 1, delay)
            down(int(position == "first") * 2)

        down(int(position != "first"))
    else:
        for i, item in enumerate(iterable):
            yield item


# Encode pil image bytes as base64 string
def encodeImage(image, format):
    if format == "png":
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()
        return base64.b64encode(image_bytes).decode("utf-8")
    else:
        return base64.b64encode(image.convert("RGBA").tobytes()).decode("utf-8")


# Decode base64 string to pil image
def decodeImage(imageString):
    try:
        if imageString["format"] == "png":
            return Image.open(BytesIO(base64.b64decode(imageString["image"]))).convert(
                "RGB"
            )
        else:
            return Image.frombytes(
                format,
                (imageString["width"], imageString["height"]),
                base64.b64decode(imageString["image"]),
            ).convert("RGB")
    except:
        rprint(
            f"\n[#ab333d]ERROR: Image cannot be decoded from bytes. It may have been corrupted."
        )
        print(imageString)
        return None


# Open the image and convert it to a tensor with values with range -1, 1
def load_img(image, h0, w0):
    image.convert("RGB")
    w, h = image.size

    # Override the image size if h0 and w0 are provided
    if h0 is not None and w0 is not None:
        h, w = h0, w0

    # Adjust the width and height to be divisible by 8 and resize the image using bicubic resampling
    w, h = map(lambda x: x - x % 8, (w, h))
    image = image.resize((w, h), resample=Image.Resampling.BICUBIC)

    # Color adjustments to account for Tiny Autoencoder
    contrast = ImageEnhance.Contrast(image)
    image_contrast = contrast.enhance(0.78)
    saturation = ImageEnhance.Color(image_contrast)
    image_saturation = saturation.enhance(0.833)

    # Convert the image to a numpy array of float32 values in the range [0, 1], transpose it, and convert it to a PyTorch tensor
    image = np.array(image_saturation).astype(np.float32) / 255
    image = rearrange(image, "h w c -> c h w")
    image = torch.from_numpy(image).unsqueeze(0)

    # Apply a normalization by scaling the values in the range [-1, 1]
    return image


# Run blip captioning for each image in a set with optional starting prompts
def caption_images(blip, images, prompt=None):
    processor = blip["processor"]
    model = blip["model"]

    outputs = []
    for image in images:
        if prompt is not None:
            inputs = processor(image, prompt, return_tensors="pt")
        else:
            inputs = processor(image, return_tensors="pt")

        outputs.append(
            processor.decode(
                model.generate(**inputs, max_new_tokens=30)[0], skip_special_tokens=True
            )
        )
    return outputs


# Flatten a model into its layers
def flatten(el):
    # Flatten nested elements by recursively traversing through children
    flattened = [flatten(children) for children in el.children()]
    res = [el]
    for c in flattened:
        res += c
    return res


# Gamma adjustment
def adjust_gamma(image, gamma=1.0):
    # Create a lookup table for the gamma function
    gamma_map = [255 * ((i / 255.0) ** (1.0 / gamma)) for i in range(256)]
    gamma_table = bytes([(int(x / 255.0 * 65535.0) >> 8) for x in gamma_map] * 3)

    # Apply the gamma correction using the lookup table
    return image.point(gamma_table)


# Load blip image captioning model
def load_blip(path):
    timer = time.time()
    print("\nLoading vision model")
    try:
        processor = BlipProcessor.from_pretrained(path)
        model = BlipForConditionalGeneration.from_pretrained(path)
        play("iteration.wav")
        rprint(f"[#c4f129]Loaded in [#48a971]{round(time.time()-timer, 2)} [#c4f129]seconds")
        return {"processor": processor, "model": model}
    except Exception as e:
        rprint(f"[#ab333d]{traceback.format_exc()}\n\nBLIP could not be loaded, this may indicate a model has not been downloaded fully, or you have run out of RAM.")
        return None


# Helper for loading model files
def load_model_from_config(model, verbose=False):
    # Load the model's state dictionary from the specified file
    try:
        # First try to load as a Safetensor, then as a pickletensor
        try:
            pl_sd = load_file(model, device="cpu")
        except:
            rprint(
                f"[#ab333d]Model is not a Safetensor. Please consider using Safetensors format for better security."
            )
            pl_sd = torch.load(model, map_location="cpu")

        sd = pl_sd

        # If "state_dict" is found in the loaded dictionary, assign it to sd
        if "state_dict" in sd:
            sd = pl_sd["state_dict"]

        return sd
    except Exception as e:
        rprint(
            f"[#ab333d]{traceback.format_exc()}\n\nThis may indicate a model has not been downloaded fully, or is corrupted."
        )


# Load stable diffusion 1.5 format model
def load_model(modelFileString, config, device, precision, optimized, split = True):
    global modelName
    global modelSettings

    modelParams = {"file": modelFileString, "device": device, "precision": precision, "optimized": optimized, "split": split}
    if modelSettings != modelParams:
        timer = time.time()

        global split_loaded
        if not split_loaded:
            unload_cldm()

        if device == "cuda" and not torch.cuda.is_available():
            if torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
                rprint(f"\n[#ab333d]GPU is not responding, loading model in CPU mode")

        global loadedDevice
        global modelType
        global modelPath
        modelPath, modelFile = os.path.split(modelFileString)
        loadedDevice = device

        # Check the modelFile and print corresponding loading message
        print()
        modelType = "pixel"
        if modelFile == "model.pxlm":
            print(f"Loading primary model")
        elif modelFile == "modelmicro.pxlm":
            print(f"Loading micro model")
        elif modelFile == "modelmini.pxlm":
            print(f"Loading mini model")
        elif modelFile == "modelmega.pxlm":
            print(f"Loading mega model")
        elif modelFile == "paletteGen.pxlm":
            modelType = "palette"
            print(f"Loading PaletteGen model")
        else:
            modelType = "general"
            rprint(f"Loading custom model from [#48a971]{modelFile}")

        # Determine if turbo mode is enabled
        turbo = True
        if optimized and device == "cuda":
            turbo = False

        # Load the model's state dictionary from the specified file
        sd = load_model_from_config(f"{os.path.join(modelPath, modelFile)}")

        # Separate the input and output blocks from the state dictionary
        if split:
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
            decoder_path = os.path.abspath("models/decoder/decoder.px")
            modelPV = load_pixelvae_model(
                decoder_path, device, "eVWtlIBjTRr0-gyZB0smWSwxCiF8l4PVJcNJOIFLFqE="
            )

        # Instantiate and load the main model
        if split:
            global model
            model = instantiate_from_config(config.model_unet)
            _, _ = model.load_state_dict(sd, strict=False)
            model.eval()
            model.unet_bs = 1
            model.cdevice = device
            model.turbo = turbo

        # Instantiate and load the conditional stage model
        global modelCS
        modelCS = instantiate_from_config(config.model_cond_stage)
        _, _ = modelCS.load_state_dict(sd, strict=False)
        modelCS.eval()
        modelCS.cond_stage_model.device = device

        # Instantiate and load the first stage model
        global modelTA
        modelTA = TAESD().to(device)

        # Set precision and device settings
        precision, fp16_mode, fp8_mode = get_precision(device, precision)
        if device == "cuda" and precision == "fp16":
            if split:
                model.to(fp16_mode)
            modelCS.to(fp16_mode)
            modelTA.to(fp16_mode)
            precision = fp16_mode
        elif device == "cuda" and precision == "fp8":
            if split:
                model.to(fp8_mode)
            for layer in flatten(modelCS):
                if isinstance(layer, torch.nn.Linear):
                    layer.to(fp8_mode)
            modelTA.to(fp16_mode)
            precision = fp8_mode
            rprint(f"Applied [#48a971]torch.fp8[/] to model")
        else:
            precision = "torch.float32"

        if split:
            assign_lora_names_to_compvis_modules(model, modelCS)

        modelName = modelFileString
        modelSettings = modelParams

        # Print loading information
        play("iteration.wav")
        rprint(f"[#c4f129]Loaded model to [#48a971]{device}[#c4f129] with [#48a971]{precision} precision[#c4f129] in [#48a971]{round(time.time()-timer, 2)} [#c4f129]seconds")
        
        if split:
            split_loaded = True
        else:
            split_loaded = False
            return sd, modelFileString


# Apply prompt enhancements, defaults, and language model management
def managePrompts(prompt, negative, W, H, seed, upscale, generations, loras, translate, promptTuning):
    timer = time.time()
    global modelLM
    global loadedDevice
    global modelType
    global sounds
    global modelPath

    prompts = [prompt] * generations

    if translate:
        # Check GPU VRAM to ensure LLM compatibility because users can't be trusted to select settings properly T-T
        cardMemory = torch.cuda.get_device_properties("cuda").total_memory / 1073741824
        if cardMemory >= 7.6:
            if cardMemory <= 10.2:
                rprint(f"\n[#494b9b]Memory is less than 10GB, image generation speed may suffer with LLM loaded.")
            try:
                # Load LLM for prompt upsampling
                if modelLM == None:
                    print("\nLoading prompt translation language model")
                    modelLM = load_chat_pipeline(os.path.join(modelPath, "LLM"))
                    play("iteration.wav")

                    rprint(f"[#c4f129]Loaded in [#48a971]{round(time.time()-timer, 2)} [#c4f129]seconds")

                if modelLM is not None:
                    try:
                        # Generate responses
                        rprint(f"\n[#48a971]Translation model [white]generating [#48a971]{generations} [white]enhanced prompts")

                        upsampled_captions = []
                        for prompt in clbar(prompts, name="Enhancing", position="", unit="prompt", prefixwidth=12, suffixwidth=28):
                            # Try to generate a response, if no response is identified after retrys, set upsampled prompt to initial prompt
                            upsampled_caption = None
                            retrys = 5
                            while upsampled_caption == None and retrys > 0:
                                outputs = upsample_caption(modelLM, prompt, seed)
                                upsampled_caption = collect_response(outputs)
                                retrys -= 1
                            seed += 1

                            if upsampled_caption == None:
                                upsampled_caption = prompt

                            upsampled_captions.append(upsampled_caption)
                            play("iteration.wav")

                        prompts = upsampled_captions
                        del outputs, upsampled_caption
                        clearCache()

                        seed = seed - len(prompts)
                        print()
                        for i, prompt in enumerate(prompts[:8]):
                            rprint(f"[#48a971]Seed: [#c4f129]{seed}[#48a971] Prompt: [#494b9b]{prompt}")
                            seed += 1
                        if len(prompts) > 8:
                            rprint(f"[#48a971]Remaining prompts generated but not displayed.")
                    except:
                        rprint(f"\n[#494b9b]Prompt enhancement failed unexpectedly. Prompts will not be edited.")
            except Exception as e:
                if "torch.cuda.OutOfMemoryError" in traceback.format_exc() or "Invalid buffer size" in traceback.format_exc():
                    rprint(f"\n[#494b9b]Translation model could not be loaded due to insufficient GPU resources.")
                elif "GPU is required" in traceback.format_exc():
                    rprint(f"\n[#494b9b]Translation model requires a GPU to be loaded.")
                else:
                    rprint(f"\n[#ab333d]ERROR:\n{traceback.format_exc()}")
                    rprint(f"\n[#494b9b]Translation model could not be loaded.")
        else:
            rprint(f"\n[#494b9b]Translation model requires a GPU with at least 8GB of VRAM. You only have {round(cardMemory)}GB.")
    else:
        if modelLM is not None:
            del modelLM
            clearCache()
            modelLM = None

    # Load lora names
    loraNames = [os.path.split(d["file"])[1] for d in loras if "file" in d]

    # Deal with prompt modifications
    if modelType == "pixel" and promptTuning:
        # Defaults
        prefix = "pixel art"
        suffix = "detailed"
        negativeList = [negative, "mutated, noise, nsfw, nude, frame, film reel, snowglobe, deformed, stock image, watermark, text, signature, username"]

        # Lora specific modifications
        if any(
            f"{_}.pxlm" in loraNames
            for _ in [
                "topdown",
                "isometric",
                "neogeo",
                "nes",
                "snes",
                "playstation",
                "gameboy",
                "gameboyadvance",
            ]
        ):
            prefix = "pixel"
            suffix = ""
        elif any(f"{_}.pxlm" in loraNames for _ in ["frontfacing", "gameicons", "flatshading"]):
            prefix = "pixel"
            suffix = "pixel art"
        elif any(f"{_}.pxlm" in loraNames for _ in ["nashorkimitems"]):
            prefix = "pixel, item"
            suffix = ""
            negativeList.insert(0, "vibrant, colorful")
        elif any(f"{_}.pxlm" in loraNames for _ in ["gamecharacters"]):
            prefix = "pixel"
            suffix = "blank background"

        if any(f"{_}.pxlm" in loraNames for _ in ["1bit"]):
            prefix = f"{prefix}, 1-bit"
            suffix = f"{suffix}, pixel art, black and white, white background"
            negativeList.insert(0, "color, colors")

        if any(f"{_}.pxlm" in loraNames for _ in ["tiling", "tiling16", "tiling32"]):
            prefix = f"{prefix}, texture"
            suffix = f"{suffix}, pixel art"

        # Model specific modifications
        if math.sqrt(W * H) >= 832 and not upscale:
            suffix = f"{suffix}, pjpixdeuc art style"

        # Combine all prompt modifications
        negatives = [", ".join(negativeList)] * generations
        for i, prompt in enumerate(prompts):
            prompts[i] = f"{prefix}, {prompt}, {suffix}"
    else:
        if promptTuning:
            negatives = [f"{negative}, pixel art, blurry, mutated, deformed, borders, watermark, text"] * generations
        else:
            negatives = [f"{negative}, pixel art"] * generations

    del loraNames
    return prompts, negatives


# K-centroid downscaling alg
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
        tile = image.crop((x * wFactor, y * hFactor, (x * wFactor) + wFactor, (y * hFactor) + hFactor))

        # Quantize the colors of the tile using k-means clustering
        tile = tile.quantize(colors=centroids, method=1, kmeans=centroids).convert("RGB")

        # Get the color counts and find the most common color
        color_counts = tile.getcolors()
        most_common_color = max(color_counts, key=lambda x: x[0])[1]

        # Assign the most common color to the corresponding pixel in the downscaled image
        downscaled[y, x, :] = most_common_color

    return Image.fromarray(downscaled, mode="RGB")


# Displays graphics for k-centroid
def kCentroidVerbose(images, width, height, centroids):
    timer = time.time()
    for i, image in enumerate(images):
        images[i] = decodeImage(image)

    rprint(f"\n[#48a971]K-Centroid downscaling[white] from [#48a971]{images[0].width}[white]x[#48a971]{images[0].height}[white] to [#48a971]{width}[white]x[#48a971]{height}[white] with [#48a971]{centroids}[white] centroids")

    # Perform k-centroid downscaling and save the image
    count = 0
    output = []
    for image in clbar(images, name = "Processed", unit = "image", prefixwidth = 12, suffixwidth = 28):
        count += 1
        resized_image = kCentroid(image, int(width), int(height), int(centroids))

        name = str(hash(str([image, width, height, centroids, count])))
        output.append({"name": name, "format": "png", "image": encodeImage(resized_image, "png")})

        if image != images[-1]:
            play("iteration.wav")
        else:
            play("batch.wav")

    rprint(f"\n[#c4f129]Resized in [#48a971]{round(time.time()-timer, 2)} [#c4f129]seconds")
    return output


# Attempts to detect the ideal pixel resolution of a given image
def pixelDetect(image: Image):
    # Thanks to https://github.com/paultron for optimizing my garbage code
    # I swapped the axis so they accurately reflect the horizontal and vertical scaling factor for images with uneven ratios

    # Convert the image to a NumPy array
    npim = np.array(image)[..., :3]

    # Compute horizontal differences between pixels
    hdiff = np.sqrt(np.sum((npim[:, :-1, :] - npim[:, 1:, :]) ** 2, axis=2))
    hsum = np.sum(hdiff, 0)

    # Compute vertical differences between pixels
    vdiff = np.sqrt(np.sum((npim[:-1, :, :] - npim[1:, :, :]) ** 2, axis=2))
    vsum = np.sum(vdiff, 1)

    # Find peaks in the horizontal and vertical sums
    hpeaks, _ = scipy.signal.find_peaks(hsum, distance=1, height=0.0)
    vpeaks, _ = scipy.signal.find_peaks(vsum, distance=1, height=0.0)

    # Compute spacing between the peaks
    hspacing = np.diff(hpeaks)
    vspacing = np.diff(vpeaks)

    # Resize input image using kCentroid with the calculated horizontal and vertical factors
    return kCentroid(image, round(image.width / np.median(hspacing)), round(image.height / np.median(vspacing)), 2)


# Displays graphics for pixelDetect
def pixelDetectVerbose(image):
    # Check if input file exists and open it
    image = decodeImage(image[0]).convert("RGB")

    rprint(f"\n[#48a971]Finding pixel ratio for current cel")

    # Process the image using pixelDetect and save the result
    for _ in clbar(range(1), name="Processed", position="last", unit="image", prefixwidth=12, suffixwidth=28):
        downscale = pixelDetect(image)

        numColors = determine_best_k(downscale, 128)

        for _ in clbar([downscale], name="Palettizing", position="first", prefixwidth=12, suffixwidth=28):
            image_indexed = downscale.quantize(colors=numColors, method=1, kmeans=numColors, dither=0).convert("RGB")

    play("batch.wav")
    return [{"name": str(hash(str(image))), "format": "png", "image": encodeImage(image_indexed, "png")}]


# Denoises an image using quantization
def kDenoise(image, smoothing, strength):
    image = image.convert("RGB")

    # Create an array to store the denoised image
    denoised = np.zeros((image.height, image.width, 3), dtype=np.uint8)

    # Iterate over each pixel
    for x, y in product(range(image.width), range(image.height)):
        # Crop the image to a 3x3 tile around the current pixel
        tile = image.crop((x - 1, y - 1, min(x + 2, image.width), min(y + 2, image.height)))

        # Calculate the number of centroids based on the tile size and strength
        centroids = max(2, min(round((tile.width * tile.height) * (1 / strength)), (tile.width * tile.height)))

        # Quantize the tile to the specified number of centroids
        tile = tile.quantize(colors=centroids, method=1, kmeans=centroids).convert("RGB")

        # Get the color counts for each centroid and find the most common color
        color_counts = tile.getcolors()
        final_color = tile.getpixel((1, 1))

        # Check if the count of the most common color is below a threshold
        count = 0
        for ele in color_counts:
            if ele[1] == final_color:
                count = ele[0]

        # If the count is below the threshold, choose the most common color
        if count < 1 + round(((tile.width * tile.height) * 0.8) * (smoothing / 10)):
            final_color = max(color_counts, key=lambda x: x[0])[1]

        # Store the final color in the downscaled image array
        denoised[y, x, :] = final_color

    return Image.fromarray(denoised, mode="RGB")


# Uses curve fitting to find the optimal number of colors to reduce an image to
def determine_best_k(image, max_k, n_samples=10000, smooth_window=7):
    image = image.convert("RGB")

    # Flatten the image pixels and sample them
    pixels = np.array(image)
    pixel_indices = np.reshape(pixels, (-1, 3))

    if pixel_indices.shape[0] > n_samples:
        pixel_indices = pixel_indices[np.random.choice(pixel_indices.shape[0], n_samples, replace=False), :]

    # Compute centroids for max_k
    quantized_image = image.quantize(colors=max_k, method=2, kmeans=max_k, dither=0)
    centroids_max_k = np.array(quantized_image.getpalette()[: max_k * 3]).reshape(-1, 3)

    distortions = []
    for k in range(1, max_k + 1):
        subset_centroids = centroids_max_k[:k]

        # Calculate distortions using SciPy
        distances = scipy.spatial.distance.cdist(pixel_indices, subset_centroids)
        min_distances = np.min(distances, axis=1)
        distortions.append(np.sum(min_distances**2))

    # Calculate slope changes
    slopes = np.diff(distortions)
    relative_slopes = np.diff(slopes) / (np.abs(slopes[:-1]) + 1e-8)

    # Find the elbow point based on the maximum relative slope change
    if len(relative_slopes) <= 1:
        return 2  # Return at least 2 if not enough data for slopes
    elbow_index = np.argmax(np.abs(relative_slopes))

    # Calculate the actual k value, considering the reductions due to diff and smoothing
    actual_k = (elbow_index + 3 + (smooth_window // 2) * 2)  # Add the reduction from diff and smoothing

    # Ensure actual_k is at least 1 and does not exceed max_k
    actual_k = max(4, min(actual_k, max_k))

    return actual_k


# Filters a list of data to remove outliers
def filterList(data, m=0.7):
    # Calculate mean of the data
    mean = sum(data) / len(data)

    # Calculate standard deviation of the data
    standard_deviation = (sum((x - mean) ** 2 for x in data) / len(data)) ** 0.5

    filtered = []
    for x in data:
        # Keep elements close to mean, replace others with a placeholder
        if abs(x - mean) < m * standard_deviation:
            filtered.append(x)
        else:
            filtered.append(9999999)
    return filtered


# Used color distances to determine the best palette to fit an image
def determine_best_palette_verbose(image, palettes):
    # Convert the image to RGB mode
    image = image.convert("RGB")

    paletteImages = []
    for palette in palettes:
        try:
            paletteImages.append(decodeImage(palette))
        except:
            pass

    # Prepare arrays for distortion calculation
    pixels = np.array(image)
    pixel_indices = np.reshape(pixels, (-1, 3))

    # Calculate distortion for different palettes
    distortions = []
    for paletteImage in clbar(paletteImages, name="Searching", position="first", prefixwidth=12, suffixwidth=28):
        palette = []

        # Extract palette colors
        palColors = paletteImage.getcolors(16777216)
        numColors = len(palColors)
        palette = np.concatenate([x[1] for x in palColors]).tolist()

        # Create a new palette image
        paletteImage = Image.new("P", (256, 1))
        paletteImage.putpalette(palette)

        quantized_image = image.quantize(method=1, kmeans=numColors, palette=paletteImage, dither=0)
        centroids = np.array(quantized_image.getpalette()[: numColors * 3]).reshape(-1, 3)

        # Calculate distortions more memory-efficiently
        min_distances = [np.min(np.linalg.norm(centroid - pixel_indices, axis=1)) for centroid in centroids]
        distortions.append(np.sum(np.square(min_distances)))

    # Find the best match
    best_match_index = np.argmin(filterList(distortions))
    return paletteImages[best_match_index], palettes[best_match_index]["name"]


# Restricts an image to a set of colors determined by the input
def palettize(images, source, paletteURL, palettes, colors, dithering, strength, denoise, smoothness, intensity):
    # Check if a palette URL is provided and try to download the palette image
    paletteImage = None
    if source == "URL":
        try:
            paletteImage = Image.open(BytesIO(requests.get(paletteURL).content)).convert("RGB")
        except:
            rprint(f"\n[#ab333d]ERROR: URL {paletteURL} cannot be reached or is not an image\nReverting to Adaptive palette")
            paletteImage = None
    elif palettes != []:
        try:
            paletteImage = decodeImage(palettes[0])
        except:
            pass

    timer = time.time()

    # Create a list to store file paths
    for i, image in enumerate(images):
        images[i] = decodeImage(image)

    # Determine the number of colors based on the palette or user input
    if paletteImage is not None:
        numColors = len(paletteImage.getcolors(16777216))
    else:
        numColors = colors

    # Create the string for conversion message
    string = (f"\n[#48a971]Converting output[white] to [#48a971]{numColors}[white] colors")

    # Add dithering information if strength and dithering are greater than 0
    if strength > 0 and dithering > 0:
        string = f"{string} with order [#48a971]{dithering}[white] dithering"

    if source == "Automatic":
        string = f"\n[#48a971]Converting output[white] to best number of colors"
    elif source == "Best Palette":
        string = f"\n[#48a971]Converting output[white] to best color palette"

    # Print the conversion message
    rprint(string)

    palFiles = []
    output = []
    count = 0
    # Process each file in the list
    for image in clbar(images, name="Processed", position="last", unit="image", prefixwidth=12, suffixwidth=28):
        # Apply denoising if enabled
        if denoise:
            image = kDenoise(image, smoothness, intensity)

        # Calculate the threshold for dithering
        threshold = 4 * strength

        if source == "Automatic":
            numColors = determine_best_k(image, 96)

        # Check if a palette file is provided
        if paletteImage is not None or source == "Best Palette":
            # Open the palette image and calculate the number of colors
            if source == "Best Palette":
                if len(palettes) > 0:
                    paletteImage, palFile = determine_best_palette_verbose(image, palettes)
                    palFiles.append(str(palFile))
                else:
                    rprint(f"\n[#ab333d]ERROR:\nNo palettes were found in the selected folder\n\n\n\n")
                    play("error.wav")
                    paletteImage = image

            numColors = len(paletteImage.getcolors(16777216))

            if strength > 0 and dithering > 0:
                for _ in clbar([image], name="Palettizing", position="first", prefixwidth=12, suffixwidth=28):
                    # Adjust the image gamma
                    image = adjust_gamma(image, 1.0 - (0.02 * strength))

                    # Extract palette colors
                    palette = [x[1] for x in paletteImage.getcolors(16777216)]

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        # Perform ordered dithering using Bayer matrix
                        palette = hitherdither.palette.Palette(palette)
                        image_indexed = hitherdither.ordered.bayer.bayer_dithering(image, palette, [threshold, threshold, threshold], order=dithering).convert("RGB")
            else:
                # Extract palette colors
                palette = np.concatenate([x[1] for x in paletteImage.getcolors(16777216)]).tolist()

                # Create a new palette image
                tempPaletteImage = Image.new("P", (256, 1))
                tempPaletteImage.putpalette(palette)

                # Perform quantization without dithering
                for _ in clbar([image], name="Palettizing", position="first", prefixwidth=12, suffixwidth=28):
                    image_indexed = image.quantize(method=1, kmeans=numColors, palette=tempPaletteImage, dither=0).convert("RGB")

        elif numColors > 0:
            if strength > 0 and dithering > 0:
                # Perform quantization with ordered dithering
                for _ in clbar([image], name="Palettizing", position="first", prefixwidth=12, suffixwidth=28):
                    image_indexed = image.quantize(colors=numColors, method=1, kmeans=numColors, dither=0).convert("RGB")

                    # Adjust the image gamma
                    image = adjust_gamma(image, 1.0 - (0.03 * strength))

                    # Extract palette colors
                    palette = [x[1] for x in image_indexed.getcolors(16777216)]

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        # Perform ordered dithering using Bayer matrix
                        palette = hitherdither.palette.Palette(palette)
                        image_indexed = hitherdither.ordered.bayer.bayer_dithering(image, palette, [threshold, threshold, threshold], order=dithering).convert("RGB")

            else:
                # Perform quantization without dithering
                for _ in clbar([image], name="Palettizing", position="first", prefixwidth=12, suffixwidth=28):
                    image_indexed = image.quantize(colors=numColors, method=1, kmeans=numColors, dither=0).convert("RGB")

        count += 1

        name = str(hash(str([count, source, paletteURL, palettes, colors, dithering, strength, denoise, smoothness, intensity])))
        output.append({"name": name, "format": "png", "image": encodeImage(image_indexed, "png")})

        if image != images[-1]:
            play("iteration.wav")
        else:
            play("batch.wav")

    rprint(f"[#c4f129]Palettized [#48a971]{len(images)}[#c4f129] images in [#48a971]{round(time.time()-timer, 2)}[#c4f129] seconds")
    if source == "Best Palette":
        rprint(f"[#c4f129]Palettes used: [#494b9b]{', '.join(palFiles)}")

    return output


# Helper function for automatic color quantization
def palettizeOutput(images):
    output = []
    # Process the image using pixelDetect and save the result
    for image in images:
        tempImage = image["image"]

        numColors = determine_best_k(tempImage, 96)

        image_indexed = tempImage.quantize(colors=numColors, method=1, kmeans=numColors, dither=0).convert("RGB")

        output.append({"name": image["name"], "seed": image["seed"], "format": image["format"], "image": image_indexed, "width": image["width"], "height": image["height"]})
    return output


# Loads and applies background segmentation model
def rembg(images, modelpath):
    timer = time.time()

    rprint(f"\n[#48a971]Removing [#48a971]{len(images)}[white] backgrounds")

    for i, image in enumerate(images):
        images[i] = decodeImage(image)

    # Process each file in the list
    count = 0
    output = []
    for image in clbar(images, name="Processed", position="", unit="image", prefixwidth=12, suffixwidth=28):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            size = math.sqrt(image.width * image.height)

            # Upscale and resize the image for segmentation
            upscale = max(1, int(1024 / size))
            resize = image.resize((image.width * upscale, image.height * upscale), resample=Image.Resampling.NEAREST)

            # Initialize and apply segmentation model
            segmenter.init(modelpath, resize.width, resize.height)
            [masked_image, mask] = segmenter.segment(resize)

            # Resize segmented image to original size and add to output
            count += 1
            masked_image = masked_image.resize((image.width, image.height), resample=Image.Resampling.NEAREST)

            name = str(hash(str([count, image])))
            output.append({"name": name, "format": "png", "image": encodeImage(masked_image, "png")})

            if image != images[-1]:
                play("iteration.wav")
            else:
                play("batch.wav")
    rprint(f"[#c4f129]Removed [#48a971]{len(images)}[#c4f129] backgrounds in [#48a971]{round(time.time()-timer, 2)}[#c4f129] seconds")
    return output


# Render image from latent usinf Tiny Autoencoder or clustered Pixel VAE
def render(modelTA, modelPV, samples_ddim, device, H, W, pixelSize, pixelvae, tilingX, tilingY, loras, post):
    if pixelvae:
        # Pixel clustering mode, lower threshold means bigger clusters
        denoise = 0.08
        x_sample = modelPV.run_cluster(samples_ddim, threshold=denoise, select="local4", wrap_x=tilingX, wrap_y=tilingY)
        # x_sample = modelPV.run_plain(samples_ddim[i:i+1])
        x_sample = x_sample[0].cpu().numpy()
    else:
        try:
            x_sample = modelTA.decoder(samples_ddim.to(device))
            x_sample = torch.clamp((x_sample.cpu().float()), min = 0.0, max = 1.0)
            x_sample = x_sample.cpu().movedim(1, -1)
            x_sample = 255.0 * x_sample[0].cpu().numpy()
            x_sample = np.clip(x_sample, 0, 255).astype(np.uint8)

            # Denoise the generated image
            x_sample = cv2.fastNlMeansDenoisingColored(x_sample, None, 6, 6, 3, 21)

            # Color adjustments to account for Tiny Autoencoder
            contrast = ImageEnhance.Contrast(Image.fromarray(x_sample))
            x_sample_contrast = contrast.enhance(1.3)
            saturation = ImageEnhance.Color(x_sample_contrast)
            x_sample_saturation = saturation.enhance(1.2)

            # Convert back to NumPy array if necessary
            x_sample = np.array(x_sample_saturation)

            
        except:
            if "torch.cuda.OutOfMemoryError" in traceback.format_exc() or "Invalid buffer size" in traceback.format_exc():
                rprint(f"\n[#ab333d]Ran out of VRAM during decode, switching to fast pixel decoder")
                # Pixel clustering mode, lower threshold means bigger clusters
                denoise = 0.08
                x_sample = modelPV.run_cluster(samples_ddim, threshold=denoise, select="local4", wrap_x=tilingX, wrap_y=tilingY)
                x_sample = x_sample[0].cpu().numpy()
            else:
                rprint(f"\n[#ab333d]ERROR:\n{traceback.format_exc()}")
    
    # Convert the numpy array to an image
    x_sample_image = Image.fromarray(x_sample.astype(np.uint8))

    if x_sample_image.width > W // pixelSize and x_sample_image.height > H // pixelSize:
        # Resize the image if pixel is true
        x_sample_image = kCentroid(x_sample_image, W // pixelSize, H // pixelSize, 2)
    elif x_sample_image.width < W // pixelSize and x_sample_image.height < H // pixelSize:
        x_sample_image = x_sample_image.resize((W, H), resample=Image.Resampling.NEAREST)

    if not pixelvae:
        # Sharpen to enhance details lost by decoding
        x_sample_image_sharp = x_sample_image.filter(ImageFilter.SHARPEN)
        alpha = 0.13
        x_sample_image = Image.blend(x_sample_image, x_sample_image_sharp, alpha)

    loraNames = [os.path.split(d["file"])[1] for d in loras if "file" in d]
    if "1bit.pxlm" in loraNames:
        post = False
        # Quantize images to 4 colors
        x_sample_image = x_sample_image.quantize(colors=4, method=1, kmeans=4, dither=0).convert('RGB')
        # Quantize images to 2 colors
        x_sample_image = x_sample_image.quantize(colors=2, method=1, kmeans=2, dither=0).convert('RGB')

        # Find the brightest color and darkest color, convert them to white and black
        pixels = list(x_sample_image.getdata())
        darkest, brightest = min(pixels), max(pixels)
        new_pixels = [0 if pixel == darkest else 255 if pixel == brightest else pixel for pixel in pixels]
        new_image = Image.new("L", x_sample_image.size)
        new_image.putdata(new_pixels)

        x_sample_image = new_image.convert('RGB')

    return x_sample_image, post


# Render image from latent using direct Pixel VAE conversion
def fastRender(modelPV, samples_ddim, pixelSize, W, H):
    x_sample = modelPV.run_plain(samples_ddim)
    x_sample = x_sample[0].cpu().numpy()
    x_sample_image = Image.fromarray(x_sample.astype(np.uint8))
    if pixelSize > 8:
        x_sample_image = x_sample_image.resize((W // pixelSize, H // pixelSize), resample=Image.Resampling.NEAREST)

    # Upscale to prevent weird Aseprite specific decode slowness for smaller images ???
    if math.sqrt(x_sample_image.width * x_sample_image.height) < 48:
        factor = math.ceil(48 / math.sqrt(x_sample_image.width * x_sample_image.height))
        x_sample_image = x_sample_image.resize((x_sample_image.width * factor, x_sample_image.height * factor), resample=Image.Resampling.NEAREST)
    return x_sample_image


# Palette generation wrapper for text to image
def paletteGen(prompt, colors, seed, device, precision):
    # Calculate the base for palette generation
    base = 2**round(math.log2(colors))

    # Calculate the width of the image based on the base and number of colors
    width = 512+((512/base)*(colors-base))

    # Generate text-to-image conversion with specified parameters
    for _ in txt2img(prompt, "", False, False, int(width), 512, 1, False, 6, 7.0, {"apply":False}, {"hue":0, "tint":0, "saturation":50, "brightness":70, "contrast":50, "outline":50}, seed, 1, 512, device, precision, [{"file": "some/path/none", "weight": 0}], False, False, False, False, False):
        image = _

    # Perform k-centroid downscaling on the image
    image = decodeImage(image["value"]["images"][0])
    image = kCentroid(image, int(image.width/(512/base)), 1, 2)

    # Iterate over the pixels in the image and set corresponding palette colors
    palette = Image.new('P', (colors, 1))
    for x in range(image.width):
        for y in range(image.height):
            r, g, b = image.getpixel((x, y))

            palette.putpixel((x, y), (r, g, b))

    name = hash(str([prompt, colors, seed, device]))
    rprint(f"[#c4f129]Image converted to color palette with [#48a971]{colors}[#c4f129] colors")
    return [{"name": f"palette{name}", "format": "png", "image": encodeImage(palette.convert("RGB"), "png")}]


def continuous_pattern_wave(x):
    """
    Continuous function that matches the given pattern:
    https://www.desmos.com/calculator/2bsi4t5gzi
    """
    # Define the pattern points and their corresponding x values
    pattern = [1, 0.5, 0, 0, 0, -0.5, -1, -0.5, 0, 0, 0, 0.5, 1]
    x_points = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    
    # Normalize x to the range of 0 to 12
    x = x % 12

    # Find the segment of the pattern that x is in
    for i in range(len(x_points) - 1):
        if x_points[i] <= x < x_points[i + 1]:
            # Perform linear interpolation
            y1, y2 = pattern[i], pattern[i + 1]
            x1, x2 = x_points[i], x_points[i + 1]
            return y1 + (y2 - y1) * (x - x1) / (x2 - x1)

    # Handle the case where x is exactly 12
    return pattern[0]


def manageComposition(lighting, composition, loras):
    lecoPath = os.path.join(modelPath, "LECO")

    if lighting["apply"]:
        left_right = (lighting["x"]-25)*8
        down_up = lighting["y"]*-6
        back_front = max(0, lighting["z"])*-3

        if down_up != 0:
            loras.append({"file": os.path.join(lecoPath, "light_du.leco"), "weight": down_up})
        if left_right != 0:
            loras.append({"file": os.path.join(lecoPath, "light_lr.leco"), "weight": left_right})
        if back_front != 0:
            loras.append({"file": os.path.join(lecoPath, "light_bf.leco"), "weight": back_front})

    hue = composition["hue"]
    tint = composition["tint"]
    cyan_red = round(continuous_pattern_wave((hue)*(12/360))*(tint*4))
    magenta_green = round(continuous_pattern_wave((hue-120)*(12/360))*(tint*4))
    yellow_blue = round(continuous_pattern_wave((hue+120)*(12/360))*(tint*4))

    if cyan_red != 0:
        loras.append({"file": os.path.join(lecoPath, "color_cr.leco"), "weight": cyan_red})
    if magenta_green != 0:
        loras.append({"file": os.path.join(lecoPath, "color_mg.leco"), "weight": magenta_green})
    if yellow_blue != 0:
        loras.append({"file": os.path.join(lecoPath, "color_yb.leco"), "weight": yellow_blue})


    # Brightness control
    brightness = sorted((0, round(composition["brightness"]), 100))[1]/10
    if brightness != 7:
        # Curve defined by https://www.desmos.com/calculator/qksu9umqae
        loras.append({"file": os.path.join(lecoPath, "brightness.leco"), "weight": round((((brightness - 13.7) ** 2) * 0.8) - 40)})

    saturation = round(composition["saturation"]-50)*8
    if saturation != 0:
        loras.append({"file": os.path.join(lecoPath, "saturation.leco"), "weight": saturation})

    contrast = round(composition["contrast"]-50)*12
    if contrast != 0:
        loras.append({"file": os.path.join(lecoPath, "contrast.leco"), "weight": contrast})

    outline = round(composition["outline"]-50)*4
    if outline != 0:
        loras.append({"file": os.path.join(lecoPath, "outline.leco"), "weight": outline})

    return loras


def prepare_inference(
    title,
    prompt,
    negative,
    translate,
    promptTuning,
    W,
    H,
    pixelSize,
    steps,
    scale,
    lighting,
    composition,
    seed,
    total_images,
    maxBatchSize,
    device,
    precision,
    loras,
    
    # options for image to image
    image = None,
):
    raw_loras = []
    
    # Check gpu availability
    if device == "cuda" and not torch.cuda.is_available():
        if torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
            rprint(f"\n[#ab333d]GPU is not responding, loading model in CPU mode")

    if image is not None:
        # Load initial image and move it to the specified device
        image = image.resize((W, H), resample=Image.Resampling.BILINEAR)
        init_image = load_img(image.convert("RGB"), H, W).to(device)

    # Calculate maximum batch size
    global maxSize
    maxSize = maxBatchSize
    size = math.sqrt(W * H)
    if size >= maxSize or device == "cpu":
        batch = 1
    else:
        batch = min(total_images, math.floor((maxSize / size) ** 2))
    runs = (
        math.floor(total_images / batch)
        if total_images % batch == 0
        else math.floor(total_images / batch) + 1
    )

    # Set the seed for random number generation if not provided
    if seed == None:
        seed = randint(0, 1000000)

    # Set attention map tile values
    wtile = max_tile(W // 8)
    htile = max_tile(H // 8)

    global modelPath
    # Composition and lighting modifications
    loras = manageComposition(lighting, composition, loras)

    lecoPath = os.path.join(modelPath, "LECO")
    found_contrast = False
    for lora in loras:
        if lora["file"] == os.path.join(lecoPath, "brightness.leco"):
            lora["weight"] = lora["weight"] - 40
        if lora["file"] == os.path.join(lecoPath, "contrast.leco"):
            found_contrast = True
            lora["weight"] = lora["weight"] + 120
    if not found_contrast:
        loras.append({"file": os.path.join(lecoPath, "contrast.leco"), "weight": 120})

    # Apply modifications to raw prompts
    data, negative_data = managePrompts(
        prompt,
        negative,
        W,
        H,
        seed,
        False,
        total_images,
        loras,
        translate,
        promptTuning,
    )
    seed_everything(seed)

    rprint(
        f"\n[#48a971]{title}[white] generating [#48a971]{total_images}[white] images with [#48a971]{steps}[white] steps over [#48a971]{runs}[white] batches with [#48a971]{wtile}[white]x[#48a971]{htile}[white] attention tiles at [#48a971]{W}[white]x[#48a971]{H}[white] ([#48a971]{W // pixelSize}[white]x[#48a971]{H // pixelSize}[white] pixels)"
    )

    global model
    global modelCS
    global modelTA
    global modelPV

    # Set the precision scope based on device and precision
    precision, fp16_mode, _ = get_precision(device, precision)
    precision_scope = autocast(device, precision, fp16_mode)

    # !!! REMEMBER: ALL MODEL FILES ARE BOUND UNDER THE LICENSE AGREEMENTS OUTLINED HERE: https://astropulse.co/#retrodiffusioneula https://astropulse.co/#retrodiffusionmodeleula !!!
    decryptedFiles = []
    fernet = Fernet("I47jl1hqUPug4KbVYd60_zeXhn_IH_ECT3QRGiBxdxo=")
    for i, loraPair in enumerate(loras):
        decryptedFiles.append("none")
        _, loraName = os.path.split(loraPair["file"])
        if loraName != "none":
            # Handle proprietary models
            if os.path.splitext(loraName)[1] == ".pxlm":
                try:
                    with open(loraPair["file"], "rb") as enc_file:
                        encrypted = enc_file.read()
                        try:
                            # Assume file is encrypted, decrypt it
                            decryptedFiles[i] = fernet.decrypt(encrypted)
                        except:
                            # Decryption failed, assume not encrypted
                            decryptedFiles[i] = encrypted

                        with open(loraPair["file"], "wb") as dec_file:
                            # Write attempted decrypted file
                            dec_file.write(decryptedFiles[i])
                            try:
                                raw_loras.append(
                                    {
                                        "sd": load_lora_raw(loraPair["file"]),
                                        "weight": loraPair["weight"],
                                    }
                                )   
                            except:
                                # Decrypted file could not be read, revert to unchanged, and return an error
                                decryptedFiles[i] = "none"
                                dec_file.write(encrypted)
                                rprint(f"[#ab333d]Modifier {os.path.splitext(loraName)[0]} could not be loaded, the file may be corrupted")
                                continue
                except:
                    rprint(f"[#ab333d]Modifier {os.path.splitext(loraName)[0]} could not be loaded, the file may be corrupted")
            else:
                # Add lora to unet
                raw_loras.append(
                    {
                        "sd": load_lora_raw(loraPair["file"]),
                        "weight": loraPair["weight"],
                    }
                )
                
            if not any(name == os.path.splitext(loraName)[0] for name in system_models):
                rprint(f"[#494b9b]Using [#48a971]{os.path.splitext(loraName)[0]} [#494b9b]LoRA with [#48a971]{loraPair['weight']}% [#494b9b]strength")

    seeds = []
    # Create conditioning values for each batch, then unload the text encoder
    negative_conditioning = []
    conditioning = []
    shape = []
    encoded_latent = []
    with precision_scope:
        if image is not None:
            latentBatch = batch
            latentCount = 0

            # Move the initial image to latent space and resize it
            init_latent_base = modelTA.encoder(init_image)
            init_latent_base = torch.nn.functional.interpolate(init_latent_base, size=(H // 8, W // 8), mode="bilinear") * 6.0
            if init_latent_base.shape[0] < latentBatch:
                # Create tiles of inputs to match batch arrangement
                init_latent_base = init_latent_base.repeat(
                    [math.ceil(latentBatch / init_latent_base.shape[0])]
                    + [1] * (len(init_latent_base.shape) - 1)
                )[:latentBatch]

            for run in range(runs):
                if total_images - latentCount < latentBatch:
                    latentBatch = total_images - latentCount

                    # Slice latents to new batch size
                    init_latent_base = init_latent_base[:latentBatch]

                # Encode the scaled latent
                encoded_latent.append(init_latent_base)
                latentCount += latentBatch
        else:
            for run in range(runs):
                encoded_latent.append(None)

        modelCS.to(device)
        condBatch = batch
        condCount = 0
        for run in range(runs):
            # Compute conditioning tokens using prompt and negative
            condBatch = min(condBatch, total_images - condCount)

            # Concepts containing attributes and weights (can contain negative attributes or not).
            #attributes = {"sliders": [{"token": "mountains", "neg_token": "lakes", "weight": 0.5}, {"token": "raven", "weight": 0.3}]}
            attributes = {"sliders": []}

            # Pull original text embedding for comparison
            text_embed = modelCS.get_learned_conditioning(data[condCount:condCount+condBatch])
            t = torch.zeros_like(text_embed)

            # Run through all attributes
            for slider in attributes["sliders"]:
                # Extract pure positive attribute
                token_embed = modelCS.get_learned_conditioning(slider["token"])

                # Check for negative attribute
                if "neg_token" in slider:
                    # Extract pure negative attribute
                    neg_token_embed = modelCS.get_learned_conditioning(slider["neg_token"])
                else:
                    # Add blank embed as negative, reduce weight to compensate
                    neg_token_embed = modelCS.get_learned_conditioning("")
                    slider['neg_token'] = f"not-{slider['token']}"

                # Calculate difference between positive and negative attributes
                diff = token_embed - neg_token_embed

                # Sort and collect only the most different weights according to variance
                concept = torch.argsort(diff[:, :5, :].abs().sum(axis=1).squeeze(0), descending=True)

                # Calculate the elbow point in the differences between concepts
                gradients = concept[:-1] - concept[1:]
                second_derivatives = gradients[:-1] - gradients[1:]
                elbow_point = (torch.argmax(torch.abs(second_derivatives)) + 1) // 4

                # Slice irrelevant weights from concept embedding
                concept = concept[:elbow_point]

                # Mask and multiply by weight
                concept_mask = torch.zeros_like(diff)
                concept_mask[:, :, concept] = slider['weight']
                diff = diff * concept_mask
                t += diff
                rprint(f"[#494b9b]Applying [#48a971]{slider['neg_token']} <-> {slider['token']} [#494b9b]attribute control with [#48a971]{round(slider['weight']*100)}% [#494b9b]strength")
            
            # Apply attributes to text embedding
            text_embed += t

            conditioning.append(text_embed)
            negative_conditioning.append(modelCS.get_learned_conditioning(negative_data[condCount : condCount + condBatch]))
            shape.append([condBatch, 4, H // 8, W // 8])
            condCount += condBatch

        # Move modelCS to CPU if necessary to free up GPU memory
        if device == "cuda":
            mem = torch.cuda.memory_allocated() / 1e6
            modelCS.to("cpu")
            # Wait until memory usage decreases
            while torch.cuda.memory_allocated() / 1e6 >= mem:
                time.sleep(1)
        return conditioning, negative_conditioning, encoded_latent, steps, scale, runs, data, negative_data, seeds, batch, raw_loras
    

# Generate image from text prompt
def txt2img(
    prompt,
    negative,
    translate,
    promptTuning,
    W,
    H,
    pixelSize,
    upscale,
    quality,
    scale,
    lighting,
    composition,
    seed,
    total_images,
    maxBatchSize,
    device,
    precision,
    loras,
    tilingX,
    tilingY,
    preview,
    pixelvae,
    post,
):
    timer = time.time()
    
    load_raw_loras = False
    raw_loras = []
    
    # Check gpu availability
    if device == "cuda" and not torch.cuda.is_available():
        if torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
            rprint(f"\n[#ab333d]GPU is not responding, loading model in CPU mode")

    # Calculate maximum batch size
    global maxSize
    maxSize = maxBatchSize
    size = math.sqrt(W * H)
    if size >= maxSize or device == "cpu":
        batch = 1
    else:
        batch = min(total_images, math.floor((maxSize / size) ** 2))
    runs = (
        math.floor(total_images / batch)
        if total_images % batch == 0
        else math.floor(total_images / batch) + 1
    )

    # Set the seed for random number generation if not provided
    if seed == None:
        seed = randint(0, 1000000)

    # Set attention map tile values
    wtile = max_tile(W // 8)
    htile = max_tile(H // 8)

    # Derive steps, cfg, lcm weight from quality setting
    global modelPath
    # Curves defined by https://www.desmos.com/calculator/kny0embnkg
    steps = round(3.4 + ((quality ** 2) / 1.5))
    scale = max(1, scale * ((1.6 + (((quality - 1.6) ** 2) / 4)) / 5))
    lcm_weight = max(1.5, 10 - (quality * 1.5))
    if lcm_weight > 0:
        loras.append({"file": os.path.join(modelPath, "quality.lcm"), "weight": round(lcm_weight*10)})

    # High resolution adjustments for consistency
    gWidth = W // 8
    gHeight = H // 8

    if gWidth >= 96 or gHeight >= 96:
        resfix_weight = round(max(4, min(((((math.sqrt(gWidth * gHeight)/10) - 10) ** 3) / 3000) + 4, 7.5)) * 10)
        loras.append({"file": os.path.join(modelPath, "resfix.lcm"), "weight": resfix_weight})

    # Composition and lighting modifications
    loras = manageComposition(lighting, composition, loras)

    # Composition enhancement settings (high res fix)
    pre_steps = steps
    up_steps = 1
    if gWidth >= 96 and gHeight >= 96 and upscale:
        lower = 50
        aspect = gWidth / gHeight
        gx = gWidth
        gy = gHeight
        # Calculate initial image size from given large image
        # Targets resolutions between 64x64 and 96x96 while respecting aspect ratios
        # Interactive example here: https://editor.p5js.org/Astropulse/full/Co7CGTAnm
        gWidth = int((lower * max(1, aspect)) + ((gy / 7) * aspect))
        gHeight = int((lower * max(1, 1 / aspect)) + ((gx / 7) * (1 / aspect)))

        # Curves defined by https://www.desmos.com/calculator/kny0embnkg
        pre_steps = round(steps * ((10 - (((quality - 1.1) ** 2) / 8)) / 10))
        up_steps = round(steps * max(0.42, ((((quality - 7.2) ** 2) / 2.5) + 3.2) / 10))
    else:
        upscale = False

    # Apply modifications to raw prompts
    data, negative_data = managePrompts(
        prompt,
        negative,
        W,
        H,
        seed,
        upscale,
        total_images,
        loras,
        translate,
        promptTuning,
    )
    seed_everything(seed)

    rprint(
        f"\n[#48a971]Text to Image[white] generating [#48a971]{total_images}[white] quality [#48a971]{quality}[white] images over [#48a971]{runs}[white] batches with [#48a971]{wtile}[white]x[#48a971]{htile}[white] attention tiles at [#48a971]{W}[white]x[#48a971]{H}[white] ([#48a971]{W // pixelSize}[white]x[#48a971]{H // pixelSize}[white] pixels)"
    )

    if W // 8 >= 96 and H // 8 >= 96 and upscale:
        rprint(
            f"[#48a971]Pre-generating[white] composition image at [#48a971]{gWidth * 8}[white]x[#48a971]{gHeight * 8} [white]([#48a971]{(gWidth * 8) // pixelSize}[white]x[#48a971]{(gHeight * 8) // pixelSize}[white] pixels)"
        )

    start_code = None
    sampler = "pxlcm"

    global model
    global modelCS
    global modelTA
    global modelPV

    # Patch tiling for model and modelTA
    if load_raw_loras == False: # ignore for CLDM
        model, modelTA, modelPV = patch_tiling(tilingX, tilingY, model, modelTA, modelPV)

    # Set the precision scope based on device and precision
    precision, fp16_mode, _ = get_precision(device, precision)
    precision_scope = autocast(device, precision, fp16_mode)

    # !!! REMEMBER: ALL MODEL FILES ARE BOUND UNDER THE LICENSE AGREEMENTS OUTLINED HERE: https://astropulse.co/#retrodiffusioneula https://astropulse.co/#retrodiffusionmodeleula !!!
    loadedLoras = []
    decryptedFiles = []
    fernet = Fernet("I47jl1hqUPug4KbVYd60_zeXhn_IH_ECT3QRGiBxdxo=")
    for i, loraPair in enumerate(loras):
        decryptedFiles.append("none")
        _, loraName = os.path.split(loraPair["file"])
        if loraName != "none":
            # Handle proprietary models
            if os.path.splitext(loraName)[1] == ".pxlm":
                with open(loraPair["file"], "rb") as enc_file:
                    encrypted = enc_file.read()
                    try:
                        # Assume file is encrypted, decrypt it
                        decryptedFiles[i] = fernet.decrypt(encrypted)
                    except:
                        # Decryption failed, assume not encrypted
                        decryptedFiles[i] = encrypted

                    with open(loraPair["file"], "wb") as dec_file:
                        # Write attempted decrypted file
                        dec_file.write(decryptedFiles[i])
                        try:
                            if load_raw_loras:
                                raw_loras.append(
                                    {
                                        "sd": load_lora_raw(loraPair["file"]),
                                        "weight": loraPair["weight"],
                                    }
                                )
                            else:
                                # Load decrypted
                                loadedLoras.append(load_lora(loraPair["file"], model))    
                        except:
                            if load_raw_loras == False:
                                # Decrypted file could not be read, revert to unchanged, and return an error
                                decryptedFiles[i] = "none"
                                dec_file.write(encrypted)
                                loadedLoras.append(None)
                                rprint(
                                    f"[#ab333d]Modifier {os.path.splitext(loraName)[0]} could not be loaded, the file may be corrupted"
                                )
                                continue
            else:
                # Add lora to unet
                if load_raw_loras == False:
                    loadedLoras.append(load_lora(loraPair["file"], model))
                
            if load_raw_loras == False:
                loadedLoras[i].multiplier = loraPair["weight"] / 100
                # Prepare for inference
                register_lora_for_inference(loadedLoras[i])
                apply_lora()
                if not any(name == os.path.splitext(loraName)[0] for name in system_models):
                    rprint(
                        f"[#494b9b]Using [#48a971]{os.path.splitext(loraName)[0]} [#494b9b]LoRA with [#48a971]{loraPair['weight']}% [#494b9b]strength"
                    )
        else:
            if load_raw_loras == False:
                loadedLoras.append(None)

    seeds = []
    # with torch.no_grad():
    # Create conditioning values for each batch, then unload the text encoder
    negative_conditioning = []
    conditioning = []
    shape = []
    # Use the specified precision scope
    with precision_scope:
        modelCS.to(device)
        condBatch = batch
        condCount = 0
        for run in range(runs):
            # Compute conditioning tokens using prompt and negative
            condBatch = min(condBatch, total_images - condCount)

            # Concepts containing attributes and weights (can contain negative attributes or not).
            #attributes = {"sliders": [{"token": "mountains", "neg_token": "lakes", "weight": 0.5}, {"token": "raven", "weight": 0.3}]}
            attributes = {"sliders": []}

            # Pull original text embedding for comparison
            text_embed = modelCS.get_learned_conditioning(data[condCount:condCount+condBatch])
            t = torch.zeros_like(text_embed)

            # Run through all attributes
            for slider in attributes["sliders"]:
                # Extract pure positive attribute
                token_embed = modelCS.get_learned_conditioning(slider["token"])

                # Check for negative attribute
                if "neg_token" in slider:
                    # Extract pure negative attribute
                    neg_token_embed = modelCS.get_learned_conditioning(slider["neg_token"])
                else:
                    # Add blank embed as negative, reduce weight to compensate
                    neg_token_embed = modelCS.get_learned_conditioning("")
                    slider['neg_token'] = f"not-{slider['token']}"

                # Calculate difference between positive and negative attributes
                diff = token_embed - neg_token_embed

                # Sort and collect only the most different weights according to variance
                concept = torch.argsort(diff[:, :5, :].abs().sum(axis=1).squeeze(0), descending=True)

                # Calculate the elbow point in the differences between concepts
                gradients = concept[:-1] - concept[1:]
                second_derivatives = gradients[:-1] - gradients[1:]
                elbow_point = (torch.argmax(torch.abs(second_derivatives)) + 1) // 4

                # Slice irrelevant weights from concept embedding
                concept = concept[:elbow_point]

                # Mask and multiply by weight
                concept_mask = torch.zeros_like(diff)
                concept_mask[:, :, concept] = slider['weight']
                diff = diff * concept_mask
                t += diff
                rprint(f"[#494b9b]Applying [#48a971]{slider['neg_token']} <-> {slider['token']} [#494b9b]attribute control with [#48a971]{round(slider['weight']*100)}% [#494b9b]strength")
            
            # Apply attributes to text embedding
            text_embed += t

            conditioning.append(text_embed)
            negative_conditioning.append(modelCS.get_learned_conditioning(negative_data[condCount : condCount + condBatch]))
            shape.append([condBatch, 4, gHeight, gWidth])
            condCount += condBatch

        # Move modelCS to CPU if necessary to free up GPU memory
        if device == "cuda":
            mem = torch.cuda.memory_allocated() / 1e6
            modelCS.to("cpu")
            # Wait until memory usage decreases
            while torch.cuda.memory_allocated() / 1e6 >= mem:
                time.sleep(1)
    
    with torch.no_grad():
        base_count = 0
        output = []

        # Iterate over the specified number of iterations
        for run in clbar(range(runs), name="Batches", position="last", unit="batch", prefixwidth=12, suffixwidth=28):
            batch = min(batch, total_images - base_count)

            # Use the specified precision scope
            with precision_scope:
                # Generate samples using the model
                for step, samples_ddim in enumerate(
                    model.sample(
                        S=pre_steps,
                        conditioning=conditioning[run],
                        seed=seed,
                        shape=shape[run],
                        verbose=False,
                        unconditional_guidance_scale=scale,
                        unconditional_conditioning=negative_conditioning[run],
                        eta=0.0,
                        x_T=start_code,
                        sampler=sampler,
                    )
                ):
                    if preview:
                        # Render and send image previews
                        displayOut = []
                        for i in range(batch):
                            x_sample_image = fastRender(modelPV, samples_ddim[i:i+1], pixelSize, W, H)
                            name = str(seed+i)
                            displayOut.append({"name": name, "seed": seed+i, "format": "bytes", "image": encodeImage(x_sample_image, "bytes"), "width": x_sample_image.width, "height": x_sample_image.height})
                        yield {
                            "action": "display_title",
                            "type": "txt2img",
                            "value": {
                                "text": f"Generating... {step}/{pre_steps} steps in batch {run+1}/{runs}"
                            },
                        }
                        yield {
                            "action": "display_image",
                            "type": "txt2img",
                            "value": {
                                "images": displayOut,
                                "prompts": data,
                                "negatives": negative_data,
                            },
                        }

                if upscale:
                    # Apply 'cropped' lora for enhanced composition at high resolution
                    crop_weight = max(3, min(round(math.sqrt(2 * ((math.sqrt((W // 8) * (H // 8))/10) - 9)), 2), 7))
                    if True:
                        loraPair = {"file": os.path.join(os.path.join(modelPath, "LECO"), "crop.leco"), "weight": crop_weight}
                        loras.append(loraPair)
                        decryptedFiles.append("none")
                        _, loraName = os.path.split(loraPair["file"])
                        loadedLoras.append(load_lora(loraPair["file"], model))
                        loadedLoras[len(loadedLoras)-1].multiplier = loraPair["weight"]
                        # Prepare for inference
                        register_lora_for_inference(loadedLoras[len(loadedLoras)-1])
                        apply_lora()

                    # Upscale latents using bilinear interpolation
                    samples_ddim = torch.nn.functional.interpolate(
                        samples_ddim, size=(H // 8, W // 8), mode="bilinear"
                    )
                    # Encode latents
                    encoded_latent = model.stochastic_encode(
                        samples_ddim,
                        torch.tensor([up_steps]).to(device),
                        seed,
                        0.0,
                        int(up_steps * 1.5),
                    )
                    # Sample for up_steps
                    for step, samples_ddim in enumerate(
                        model.sample(
                            up_steps,
                            conditioning[run],
                            encoded_latent,
                            unconditional_guidance_scale=scale,
                            unconditional_conditioning=negative_conditioning[run],
                            sampler="ddim",
                        )
                    ):
                        if preview:
                            # Render and send image previews
                            displayOut = []
                            for i in range(batch):
                                x_sample_image = fastRender(modelPV, samples_ddim[i:i+1], pixelSize, W, H)
                                name = str(seed+i)
                                displayOut.append({"name": name, "seed": seed+i, "format": "bytes", "image": encodeImage(x_sample_image, "bytes"), "width": x_sample_image.width, "height": x_sample_image.height})
                            yield {
                                "action": "display_title",
                                "type": "txt2img",
                                "value": {
                                    "text": f"Generating... {step}/{up_steps} steps in batch {run+1}/{runs}"
                                },
                            }
                            yield {
                                "action": "display_image",
                                "type": "txt2img",
                                "value": {
                                    "images": displayOut,
                                    "prompts": data,
                                    "negatives": negative_data,
                                },
                            }

                # Render final images in batch
                for i in range(batch):
                    x_sample_image, post = render(
                        modelTA,
                        modelPV,
                        samples_ddim[i:i+1],
                        device,
                        H,
                        W,
                        pixelSize,
                        pixelvae,
                        tilingX,
                        tilingY,
                        loras,
                        post,
                    )

                    if total_images > 1 and (base_count + 1) < total_images:
                        play("iteration.wav")

                    seeds.append(str(seed))

                    name = str(hash(str([data[i], negative_data[i], translate, promptTuning, W, H, upscale, quality, scale, device, loras, tilingX, tilingY, pixelvae, seed, post])) & 0x7FFFFFFFFFFFFFFF)
                    output.append({"name": name, "seed": seed, "format": "png", "image": x_sample_image, "width": x_sample_image.width, "height": x_sample_image.height})

                    seed += 1
                    base_count += 1
                # Delete the samples to free up memory
                del samples_ddim

        with precision_scope:
            for i, lora in enumerate(loadedLoras):
                if lora is not None:
                    # Release lora
                    remove_lora_for_inference(lora)
                if os.path.splitext(loras[i]["file"])[1] == ".pxlm":
                    if decryptedFiles[i] != "none":
                        encrypted = fernet.encrypt(decryptedFiles[i])
                        with open(loras[i]["file"], "wb") as dec_file:
                            dec_file.write(encrypted)
        del loadedLoras

        if post:
            output = palettizeOutput(output)

        final = []
        for image in output:
            final.append({"name": image["name"], "seed": image["seed"], "format": image["format"], "image": encodeImage(image["image"], "png"), "width": image["width"], "height": image["height"]})
        play("batch.wav")
        rprint(
            f"[#c4f129]Image generation completed in [#48a971]{round(time.time()-timer, 2)} [#c4f129]seconds\n[#48a971]Seeds: [#494b9b]{', '.join(seeds)}"
        )
        yield {
            "action": "display_image",
            "type": "txt2img",
            "value": {"images": final, "prompts": data, "negatives": negative_data},
        }


def resize_image(original_width, original_height, target_size = 512):
    target_area = target_size ** 2
    aspect_ratio = original_width / original_height
    
    new_width = math.sqrt(target_area * aspect_ratio)
    new_height = math.sqrt(target_area / aspect_ratio)
    
    # Adjust dimensions if they exceed the original dimensions, maintaining aspect ratio
    if new_width > original_width or new_height > original_height:
        if original_width > original_height:
            new_width = original_width
            new_height = new_width / aspect_ratio
        else:
            new_height = original_height
            new_width = new_height * aspect_ratio
    
    return (int(new_width), int(new_height))  # Returning integer values for dimensions


def neural_inference(modelFileString, title, controlnets, prompt, negative, autocaption, translate, promptTuning, W, H, pixelSize, steps, scale, strength, lighting, composition, seed, total_images, maxBatchSize, device, precision, loras, preview, pixelvae, mapColors, post, init_img = None):
    timer = time.time()
    global modelCS
    global modelTA
    global modelPV

    if autocaption and init_img is not None:
        global modelBLIP
        if modelBLIP is None:
            global modelPath
            modelBLIP = load_blip(os.path.join(modelPath, "BLIP"))

        if modelBLIP is not None:
            processor = modelBLIP["processor"]
            model = modelBLIP["model"]

            blip_image = init_img.resize(resize_image(init_img.width, init_img.height, 512), resample=Image.Resampling.BILINEAR)
            if prompt is not None:
                inputs = processor(blip_image, prompt, return_tensors="pt")
            else:
                inputs = processor(blip_image, return_tensors="pt")

            rprint(f"\n[#48a971]Vision model [/]generating image description")
            prompt = remove_repeated_words(processor.decode(model.generate(**inputs, max_new_tokens=30)[0], skip_special_tokens=True))
            rprint(f"[#48a971]Caption: [#494b9b]{prompt}")
    
    conditioning, negative_conditioning, image_embed, steps, scale, runs, data, negative_data, seeds, batch, raw_loras = prepare_inference(
        title, prompt, negative, translate, promptTuning, W, H, pixelSize, steps, scale, lighting, composition, seed, total_images, maxBatchSize, device, precision, loras, init_img)

    title = title.lower().replace(' ', '_')

    rprint(f"[#48a971]Patching model for controlnet")
    model_patcher, cldm_cond, cldm_uncond = load_controlnet(
        controlnets,
        W,
        H,
        modelFileString,
        0, # might need to point to the physical device, in this case defaults to first GPU available
        conditioning,
        negative_conditioning,
        loras = raw_loras
    )

    _, fp16_mode, _ = get_precision(device, precision)

    with torch.no_grad():
        base_count = 0
        output = []

        # Iterate over the specified number of iterations
        for run in clbar(range(runs), name="Batches", position="last", unit="batch", prefixwidth=12, suffixwidth=28):
            batch = min(batch, total_images - base_count)

            for step, samples_ddim in enumerate(sample_cldm(
                model_patcher,
                cldm_cond,
                cldm_uncond,
                seed,
                steps, # steps,
                scale + 2.0, # cfg,
                "ddim", # sampler,
                batch, # batch size
                W,
                H,
                image_embed[run], # initial latent for img2img
                strength, # denoise strength
                "normal" # scheduler
            )):
                samples_ddim = samples_ddim.to(fp16_mode)
                if preview:
                    # Render and send image previews
                    displayOut = []
                    for i in range(batch):
                        x_sample_image = fastRender(modelPV, samples_ddim[i:i+1], pixelSize, W, H)
                        name = str(seed+i)
                        displayOut.append({"name": name, "seed": seed+i, "format": "bytes", "image": encodeImage(x_sample_image, "bytes"), "width": x_sample_image.width, "height": x_sample_image.height})
                    yield {
                        "action": "display_title",
                        "type": title,
                        "value": {
                            "text": f"Generating... {step}/{steps} steps in batch {run+1}/{runs}"
                        },
                    }
                    yield {
                        "action": "display_image",
                        "type": title,
                        "value": {
                            "images": displayOut,
                            "prompts": data,
                            "negatives": negative_data,
                        },
                    }
            
            for i in range(batch):
                x_sample_image, post = render(
                    modelTA,
                    modelPV,
                    samples_ddim[i:i+1],
                    device,
                    H,
                    W,
                    pixelSize,
                    pixelvae,
                    False,
                    False,
                    raw_loras,
                    post,
                )
                if total_images > 1 and (base_count + 1) < total_images:
                    play("iteration.wav")

                seeds.append(str(seed))
                name = [data[i], negative_data[i], translate, promptTuning, W, H, steps, scale, device, loras, pixelvae, seed]
                if init_img is not None:
                    name.append(init_img.resize((16, 16), resample=Image.Resampling.NEAREST))
                name = str(hash(str(name)) & 0x7FFFFFFFFFFFFFFF)
                output.append({"name": name, "seed": seed, "format": "png", "image": x_sample_image, "width": x_sample_image.width, "height": x_sample_image.height})

                seed += 1
                base_count += 1
            # Delete the samples to free up memory
            del samples_ddim

        if mapColors and init_img is not None:
            numColors = 256
            palette_img = init_img.resize((W // 8, H // 8), resample=Image.Resampling.NEAREST)
            palette_img = palette_img.quantize(colors=numColors, method=1, kmeans=numColors, dither=0).convert("RGB")
            numColors = len(palette_img.getcolors(numColors))

            # Extract palette colors
            palette = np.concatenate([x[1] for x in palette_img.getcolors(numColors)]).tolist()

            # Create a new palette image
            tempPaletteImage = Image.new("P", (256, 1))
            tempPaletteImage.putpalette(palette)

            # Convert generated image to reduced input image palette
            temp_output = output
            output = []
            for image in temp_output:
                tempImage = image["image"]
                # Perform quantization without dithering
                image_indexed = tempImage.quantize(method=1, kmeans=numColors, palette=tempPaletteImage, dither=0).convert("RGB")
                if post:
                    numColors = determine_best_k(image_indexed, 96)
                    image_indexed = image_indexed.quantize(colors=numColors, method=1, kmeans=numColors, dither=0).convert("RGB")

                output.append({"name": image["name"], "seed": image["seed"], "format": image["format"], "image": image_indexed, "width": image["width"], "height": image["height"]})
        elif post:
            output = palettizeOutput(output)

        final = []
        for image in output:
            final.append({"name": image["name"], "seed": image["seed"], "format": image["format"], "image": encodeImage(image["image"], "png"), "width": image["width"], "height": image["height"]})
        play("batch.wav")
        rprint(
            f"[#c4f129]Image generation completed in [#48a971]{round(time.time()-timer, 2)} [#c4f129]seconds\n[#48a971]Seeds: [#494b9b]{', '.join(seeds)}"
        )
        yield {
            "action": "display_image",
            "type": title,
            "value": {"images": final, "prompts": data, "negatives": negative_data},
        }

        unload_cldm()


# Generate image from image+text prompt
def img2img(
    prompt,
    negative,
    translate,
    promptTuning,
    W,
    H,
    pixelSize,
    quality,
    scale,
    strength,
    lighting,
    composition,
    seed,
    total_images,
    maxBatchSize,
    device,
    precision,
    loras,
    images,
    tilingX,
    tilingY,
    preview,
    pixelvae,
    post,
):
    timer = time.time()

    # Check gpu availability
    if device == "cuda" and not torch.cuda.is_available():
        if torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
            rprint(f"\n[#ab333d]GPU is not responding, loading model in CPU mode")

    # Load initial image and move it to the specified device
    init_img = decodeImage(images[0])
    init_image = load_img(init_img, H, W).to(device)

    # Calculate maximum batch size
    global maxSize
    maxSize = maxBatchSize
    size = math.sqrt(W * H)
    if size >= maxSize or device == "cpu":
        batch = 1
    else:
        batch = min(total_images, math.floor((maxSize / size) ** 2))
    runs = (
        math.floor(total_images / batch)
        if total_images % batch == 0
        else math.floor(total_images / batch) + 1
    )

    # Set the seed for random number generation if not provided
    if seed == None:
        seed = randint(0, 1000000)

    # Set attention map tile values
    wtile = max_tile(W // 8)
    htile = max_tile(H // 8)

    strength = strength / 100

    # Derive steps, cfg, lcm weight from quality setting
    global modelPath
    # Curves defined by https://www.desmos.com/calculator/kny0embnkg
    steps = round(9 + (((quality-1.85) ** 2) * 1.1))
    scale = max(1, scale * ((1.6 + (((quality - 1.6) ** 2) / 4)) / 5))
    lcm_weight = max(1.5, 10 - (quality * 1.5))
    if lcm_weight > 0:
        loras.append({"file": os.path.join(modelPath, "quality.lcm"), "weight": round(lcm_weight*10)})

    # Composition and lighting modifications
    loras = manageComposition(lighting, composition, loras)

    # High resolution adjustments for consistency
    if W // 8 >= 96 or H // 8 >= 96:
        loras.append({"file": os.path.join(modelPath, "resfix.lcm"), "weight": 40})
    
        # Apply 'cropped' lora for enhanced composition at high resolution
        crop_weight = max(3, min(round(math.sqrt(2 * ((math.sqrt((W // 8) * (H // 8))/10) - 9)) + 1, 2), 7))
        if True:
            loras.append({"file": os.path.join(os.path.join(modelPath, "LECO"), "crop.leco"), "weight": crop_weight})

    # Apply modifications to raw prompts
    data, negative_data = managePrompts(
        prompt,
        negative,
        W,
        H,
        seed,
        False,
        total_images,
        loras,
        translate,
        promptTuning,
    )
    seed_everything(seed)

    rprint(
        f"\n[#48a971]Image to Image[white] generating [#48a971]{total_images}[white] quality [#48a971]{quality}[white] images over [#48a971]{runs}[white] batches with [#48a971]{wtile}[white]x[#48a971]{htile}[white] attention tiles at [#48a971]{W}[white]x[#48a971]{H}[white] ([#48a971]{W // pixelSize}[white]x[#48a971]{H // pixelSize}[white] pixels)"
    )

    sampler = "ddim"

    global model
    global modelCS
    global modelTA
    global modelPV

    # Patch tiling for model and modelTA
    model, modelTA, modelPV = patch_tiling(tilingX, tilingY, model, modelTA, modelPV)

    # Set the precision scope based on device and precision
    precision, fp16_mode, _ = get_precision(device, precision)
    precision_scope = autocast(device, precision, fp16_mode)

    # !!! REMEMBER: ALL MODEL FILES ARE BOUND UNDER THE LICENSE AGREEMENTS OUTLINED HERE: https://astropulse.co/#retrodiffusioneula https://astropulse.co/#retrodiffusionmodeleula !!!
    loadedLoras = []
    decryptedFiles = []
    fernet = Fernet("I47jl1hqUPug4KbVYd60_zeXhn_IH_ECT3QRGiBxdxo=")
    for i, loraPair in enumerate(loras):
        decryptedFiles.append("none")
        _, loraName = os.path.split(loraPair["file"])
        if loraName != "none":
            # Handle proprietary models
            if os.path.splitext(loraName)[1] == ".pxlm":
                with open(loraPair["file"], "rb") as enc_file:
                    encrypted = enc_file.read()
                    try:
                        # Assume file is encrypted, decrypt it
                        decryptedFiles[i] = fernet.decrypt(encrypted)
                    except:
                        # Decryption failed, assume not encrypted
                        decryptedFiles[i] = encrypted

                    with open(loraPair["file"], "wb") as dec_file:
                        # Write attempted decrypted file
                        dec_file.write(decryptedFiles[i])
                        try:
                            # Load decrypted file
                            loadedLoras.append(load_lora(loraPair["file"], model))
                        except:
                            # Decrypted file could not be read, revert to unchanged, and return an error
                            decryptedFiles[i] = "none"
                            dec_file.write(encrypted)
                            loadedLoras.append(None)
                            rprint(
                                f"[#ab333d]Modifier {os.path.splitext(loraName)[0]} could not be loaded, the file may be corrupted"
                            )
                            continue
            else:
                # Add lora to unet
                loadedLoras.append(load_lora(loraPair["file"], model))
            loadedLoras[i].multiplier = loraPair["weight"] / 100
            # Prepare for inference
            register_lora_for_inference(loadedLoras[i])
            apply_lora()
            if not any(name == os.path.splitext(loraName)[0] for name in system_models):
                rprint(
                    f"[#494b9b]Using [#48a971]{os.path.splitext(loraName)[0]} [#494b9b]LoRA with [#48a971]{loraPair['weight']}% [#494b9b]strength"
                )
        else:
            loadedLoras.append(None)

    seeds = []
    strength = max(0.001, min(strength, 1.0))

    with torch.no_grad():
        # Create conditioning values for each batch, then unload the text encoder
        negative_conditioning = []
        conditioning = []
        encoded_latent = []

        with precision_scope:
            # Move the modelTA to the specified device
            # modelTA.to(device)
            latentBatch = batch
            latentCount = 0

            # Move the initial image to latent space and resize it
            init_latent_base = modelTA.encoder(init_image)
            init_latent_base = torch.nn.functional.interpolate(
                init_latent_base, size=(H // 8, W // 8), mode="bilinear"
            )
            if init_latent_base.shape[0] < latentBatch:
                # Create tiles of inputs to match batch arrangement
                init_latent_base = init_latent_base.repeat(
                    [math.ceil(latentBatch / init_latent_base.shape[0])]
                    + [1] * (len(init_latent_base.shape) - 1)
                )[:latentBatch]

            for run in range(runs):
                if total_images - latentCount < latentBatch:
                    latentBatch = total_images - latentCount

                    # Slice latents to new batch size
                    init_latent_base = init_latent_base[:latentBatch]

                # Encode the scaled latent
                encoded_latent.append(
                    model.stochastic_encode(
                        init_latent_base,
                        torch.tensor([steps]).to(device),
                        seed + (run * latentCount),
                        0.0,
                        max(steps + 1, int(steps / strength)),
                    )
                )
                latentCount += latentBatch

            modelCS.to(device)
            condBatch = batch
            condCount = 0
            for run in range(runs):
                # Compute conditioning tokens using prompt and negative
                condBatch = min(condBatch, total_images - condCount)
                negative_conditioning.append(
                    modelCS.get_learned_conditioning(
                        negative_data[condCount : condCount + condBatch]
                    )
                )
                conditioning.append(
                    modelCS.get_learned_conditioning(
                        data[condCount : condCount + condBatch]
                    )
                )
                condCount += condBatch

            # Move modelCS to CPU if necessary to free up GPU memory
            if device == "cuda":
                mem = torch.cuda.memory_allocated() / 1e6
                modelCS.to("cpu")
                # Wait until memory usage decreases
                while torch.cuda.memory_allocated() / 1e6 >= mem:
                    time.sleep(1)

        base_count = 0
        output = []
        # Iterate over the specified number of iterations
        for run in clbar(
            range(runs),
            name="Batches",
            position="last",
            unit="batch",
            prefixwidth=12,
            suffixwidth=28,
        ):
            batch = min(batch, total_images - base_count)

            # Use the specified precision scope
            with precision_scope:
                # Generate samples using the model
                for step, samples_ddim in enumerate(
                    model.sample(
                        steps,
                        conditioning[run],
                        encoded_latent[run],
                        unconditional_guidance_scale=scale,
                        unconditional_conditioning=negative_conditioning[run],
                        sampler=sampler,
                    )
                ):
                    if preview:
                        # Render and send image previews
                        displayOut = []
                        for i in range(batch):
                            x_sample_image = fastRender(modelPV, samples_ddim[i:i+1], pixelSize, W, H)
                            name = str(seed+i)
                            displayOut.append({"name": name, "seed": seed+i, "format": "bytes", "image": encodeImage(x_sample_image, "bytes"), "width": x_sample_image.width, "height": x_sample_image.height})
                        yield {
                            "action": "display_title",
                            "type": "img2img",
                            "value": {
                                "text": f"Generating... {step}/{steps} steps in batch {run+1}/{runs}"
                            },
                        }
                        yield {
                            "action": "display_image",
                            "type": "img2img",
                            "value": {
                                "images": displayOut,
                                "prompts": data,
                                "negatives": negative_data,
                            },
                        }

                # Render final images in batch
                for i in range(batch):
                    x_sample_image, post = render(
                        modelTA,
                        modelPV,
                        samples_ddim[i:i+1],
                        device,
                        H,
                        W,
                        pixelSize,
                        pixelvae,
                        tilingX,
                        tilingY,
                        loras,
                        post,
                    )

                    if total_images > 1 and (base_count + 1) < total_images:
                        play("iteration.wav")

                    seeds.append(str(seed))
                    name = str(hash(str([data[i], negative_data[i], init_img.resize((16, 16), resample=Image.Resampling.NEAREST), strength, translate, promptTuning, W, H, quality, scale, device, loras, tilingX, tilingY, pixelvae, seed, post])) & 0x7FFFFFFFFFFFFFFF)
                    output.append({"name": name, "seed": seed, "format": "png", "image": x_sample_image, "width": x_sample_image.width, "height": x_sample_image.height})

                    seed += 1
                    base_count += 1
                # Delete the samples to free up memory
                del samples_ddim

        with precision_scope:
            for i, lora in enumerate(loadedLoras):
                if lora is not None:
                    # Release lora
                    remove_lora_for_inference(lora)
                if os.path.splitext(loras[i]["file"])[1] == ".pxlm":
                    if decryptedFiles[i] != "none":
                        encrypted = fernet.encrypt(decryptedFiles[i])
                        with open(loras[i]["file"], "wb") as dec_file:
                            dec_file.write(encrypted)
        del loadedLoras

        if post:
            output = palettizeOutput(output)

        final = []
        for image in output:
            final.append({"name": image["name"], "seed": image["seed"], "format": image["format"], "image": encodeImage(image["image"], "png"), "width": image["width"], "height": image["height"]})
        play("batch.wav")
        rprint(
            f"[#c4f129]Image generation completed in [#48a971]{round(time.time()-timer, 2)} seconds\n[#48a971]Seeds: [#494b9b]{', '.join(seeds)}"
        )
        yield {
            "action": "display_image",
            "type": "img2img",
            "value": {"images": final, "prompts": data, "negatives": negative_data},
        }


# Wrapper for prompt manager
def prompt2prompt(path, prompt, negative, generations, seed):
    timer = time.time()
    global modelLM
    global sounds
    global modelPath
    modelPath = path

    prompts = [prompt] * generations
    seeds = []

    try:
        # Load LLM for prompt upsampling
        if modelLM == None:
            print("\nLoading prompt translation language model")
            modelLM = load_chat_pipeline(os.path.join(modelPath, "LLM"))
            play("iteration.wav")

            rprint(
                f"[#c4f129]Loaded in [#48a971]{round(time.time()-timer, 2)} [#c4f129]seconds"
            )
    except Exception as e:
        if "torch.cuda.OutOfMemoryError" in traceback.format_exc():
            rprint(
                f"\n[#494b9b]Translation model could not be loaded due to insufficient GPU resources."
            )
        else:
            rprint(f"\n[#494b9b]Translation model could not be loaded.")
    try:
        # Generate responses
        rprint(
            f"\n[#48a971]Translation model [white]generating [#48a971]{generations} [white]enhanced prompts"
        )

        upsampled_captions = []
        count = 0
        for prompt in clbar(
            prompts,
            name="Enhancing",
            position="",
            unit="prompt",
            prefixwidth=12,
            suffixwidth=28,
        ):
            # Try to generate a response, if no response is identified after retrys, set upsampled prompt to initial prompt
            upsampled_caption = None
            retrys = 5
            while upsampled_caption == None and retrys > 0:
                outputs = upsample_caption(modelLM, prompt, seed)
                upsampled_caption = collect_response(outputs)
                retrys -= 1
            seeds.append(str(seed))
            seed += 1
            count += 1

            if upsampled_caption == None:
                upsampled_caption = prompt

            upsampled_captions.append(upsampled_caption)
            if generations > 1 and count < generations:
                play("iteration.wav")

        prompts = upsampled_captions

        cardMemory = torch.cuda.get_device_properties("cuda").total_memory / 1073741824
        usedMemory = cardMemory - (torch.cuda.mem_get_info()[0] / 1073741824)

        if cardMemory - usedMemory < 4:
            del modelLM
            clearCache()
            modelLM = None
        else:
            clearCache()
    except:
        rprint(
            f"[#494b9b]Prompt enhancement failed unexpectedly. Prompts will not be edited."
        )

    play("batch.wav")
    rprint(
        f"[#c4f129]Prompt enhancement completed in [#48a971]{round(time.time()-timer, 2)} [#c4f129]seconds\n[#48a971]Seeds: [#494b9b]{', '.join(seeds)}"
    )

    return prompts


# Test largest image generation possible
def benchmark(device, precision, timeLimit, maxTestSize, errorRange, pixelvae, seed):
    timer = time.time()

    if device == "cuda" and not torch.cuda.is_available():
        if torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
            rprint(f"\n[#ab333d]GPU is not responding, loading model in CPU mode")

    global maxSize

    testSize = maxTestSize
    resize = testSize

    steps = 1

    tested = []

    tests = round(math.log2(maxTestSize) - math.log2(errorRange)) + 1

    # Set the seed for random number generation if not provided
    if seed == None:
        seed = randint(0, 1000000)
    seed_everything(seed)

    rprint(
        f"\n[#48a971]Running benchmark[white] with a maximum generation size of [#48a971]{maxTestSize*8}[white]x[#48a971]{maxTestSize*8}[white] ([#48a971]{maxTestSize}[white]x[#48a971]{maxTestSize}[white] pixels) for [#48a971]{tests}[white] total tests"
    )

    start_code = None
    sampler = "pxlcm"

    global model
    global modelCS
    global modelTA
    global modelPV

    # Set the precision scope based on device and precision
    precision, fp16_mode, _ = get_precision(device, precision)
    precision_scope = autocast(device, precision, fp16_mode)

    data = [""]
    negative_data = [""]
    with torch.no_grad():
        base_count = 0
        lower = 0

        # Load text encoder
        modelCS.to(device)
        uc = None
        uc = modelCS.get_learned_conditioning(negative_data)

        c = modelCS.get_learned_conditioning(data)

        # Move modelCS to CPU if necessary to free up GPU memory
        if device == "cuda":
            mem = torch.cuda.memory_allocated() / 1e6
            modelCS.to("cpu")
            # Wait until memory usage decreases
            while torch.cuda.memory_allocated() / 1e6 >= mem:
                time.sleep(1)

        # Iterate over the specified number of iterations
        for n in clbar(
            range(tests),
            name="Tests",
            position="last",
            unit="test",
            prefixwidth=12,
            suffixwidth=28,
        ):
            benchTimer = time.time()
            timerPerStep = 1
            passedTest = False
            # Use the specified precision scope
            with precision_scope:
                try:
                    shape = [1, 4, testSize, testSize]

                    # Generate samples using the model
                    for step, samples_ddim in enumerate(
                        model.sample(
                            S=steps,
                            conditioning=c,
                            seed=seed,
                            shape=shape,
                            verbose=False,
                            unconditional_guidance_scale=5.0,
                            unconditional_conditioning=uc,
                            eta=0.0,
                            x_T=start_code,
                            sampler=sampler,
                        )
                    ):
                        pass

                    render(
                        modelTA,
                        modelPV,
                        samples_ddim,
                        0,
                        device,
                        testSize,
                        testSize,
                        8,
                        pixelvae,
                        False,
                        False,
                        ["None"],
                        False,
                    )

                    # Delete the samples to free up memory
                    del samples_ddim

                    timerPerStep = round(time.time() - benchTimer, 2)

                    passedTest = True
                except:
                    passedTest = False

                if tests > 1 and (base_count + 1) < tests:
                    play("iteration.wav")

                base_count += 1

                torch.cuda.empty_cache()
                if torch.backends.mps.is_available() and device != "cpu":
                    torch.mps.empty_cache()

                if passedTest and timerPerStep <= timeLimit:
                    maxSize = testSize
                    tested.append((testSize, "[#c4f129]Passed"))
                    if n == 0:
                        rprint(f"\n[#c4f129]Maximum test size passed")
                        break
                    lower = testSize
                    testSize = round(lower + (resize / 2))
                    resize = testSize - lower
                else:
                    tested.append((testSize, "[#ab333d]Failed"))
                    testSize = round(lower + (resize / 2))
                    resize = testSize - lower
        sortedTests = sorted(tested, key=lambda x: (-x[0], x[1]))
        printTests = f"[#48a971]{sortedTests[0][0]}[white]: {sortedTests[0][1]}[white]"
        for test in sortedTests[1:]:
            printTests = f"{printTests}, [#48a971]{test[0]}[white]: {test[1]}[white]"
            if test[1] == "Passed":
                break
        play("batch.wav")
        rprint(
            f"[#c4f129]Benchmark completed in [#48a971]{round(time.time()-timer, 2)} [#c4f129]seconds\n{printTests}\n[white]The maximum size possible on your hardware with less than [#48a971]{timeLimit}[white] seconds per step is [#48a971]{maxSize*8}[white]x[#48a971]{maxSize*8}[white] ([#48a971]{maxSize}[white]x[#48a971]{maxSize}[white] pixels)"
        )


async def server(websocket):
    background = False
    try:
        assert sys.version_info >= (3, 10)
        async for message in websocket:
            # For debugging
            # print(message)
            try:
                message = json.loads(message)
                match message["action"]:
                    case "transform":
                        try:
                            title = "Neural Transform"

                            # Extract parameters from the message
                            values = message["value"]
                            modelData = values["model"]

                            if values["send_progress"]:
                                await websocket.send(
                                    json.dumps(
                                        {
                                            "action": "display_title",
                                            "type": title.lower().replace(' ', '_'),
                                            "value": {"text": "Loading model"},
                                        }
                                    )
                                )
                            
                            load_model(
                                modelData["file"],
                                "scripts/v1-inference.yaml",
                                modelData["device"],
                                modelData["precision"],
                                modelData["optimized"],
                                False
                            )
                            if values["send_progress"]:
                                await websocket.send(
                                    json.dumps(
                                        {
                                            "action": "display_title",
                                            "type": title.lower().replace(' ', '_'),
                                            "value": {"text": "Generating..."},
                                        }
                                    )
                                )
                            
                            # Neural detail workflow
                            
                            # Decode input image
                            init_img = decodeImage(values["images"][0])

                            # Resize image to output dimensions
                            image = init_img.resize((values["width"], values["height"]), resample=Image.Resampling.NEAREST).convert("RGB")

                            # Blur filter for detail
                            image_blur = image.filter(ImageFilter.BoxBlur(4))

                            # Net models, images, and weights in order
                            modelPath, _ = os.path.split(modelData["file"])
                            netPath = os.path.join(modelPath, "CONTROLNET")
                            lecoPath = os.path.join(modelPath, "LECO")
                            loras = values["loras"]
                            loras.append({"file": os.path.join(lecoPath, "detail.leco"), "weight": -50})
                            controlnets = [{"model_file": os.path.join(netPath, "Composition.safetensors"), "image": image, "weight": 1.0}]

                            for result in neural_inference(
                                modelData["file"],
                                title,
                                controlnets,
                                values["prompt"],
                                values["negative"],
                                False,
                                values["translate"],
                                values["prompt_tuning"],
                                values["width"],
                                values["height"],
                                values["pixel_size"],
                                values["steps"],
                                values["scale"],
                                1.0,
                                values["lighting"],
                                values["composition"],
                                values["seed"],
                                values["generations"],
                                values["max_batch_size"],
                                modelData["device"],
                                modelData["precision"],
                                values["loras"],
                                values["send_progress"],
                                values["use_pixelvae"],
                                False,
                                values["post_process"],
                                init_img
                            ):
                                if values["send_progress"]:
                                    await websocket.send(json.dumps(result))

                            await websocket.send(
                                json.dumps(
                                    {
                                        "action": "returning",
                                        "type": "img2img",
                                        "value": {"images": result["value"]["images"]},
                                    }
                                )
                            )
                        except Exception as e:
                            if "SSLCertVerificationError" in traceback.format_exc():
                                rprint(
                                    f"\n[#ab333d]ERROR: Latent Diffusion Model download failed due to SSL certificate error. Please run 'open /Applications/Python*/Install\ Certificates.command' in a new terminal"
                                )
                            elif (
                                "torch.cuda.OutOfMemoryError" in traceback.format_exc()
                            ):
                                rprint(
                                    f"\n[#ab333d]ERROR: Generation failed due to insufficient GPU resources. If you are running other GPU heavy programs try closing them. Also try lowering the image generation size or maximum batch size"
                                )
                                if modelLM is not None:
                                    rprint(
                                        f"\n[#ab333d]Try disabling LLM enhanced prompts to free up gpu resources"
                                    )
                            else:
                                rprint(f"\n[#ab333d]ERROR:\n{traceback.format_exc()}")
                            play("error.wav")
                            await websocket.send(json.dumps({"action": "error"}))
                    case "detail":
                        try:
                            title = "Neural Detail"

                            # Extract parameters from the message
                            values = message["value"]
                            modelData = values["model"]

                            if values["send_progress"]:
                                await websocket.send(
                                    json.dumps(
                                        {
                                            "action": "display_title",
                                            "type": title.lower().replace(' ', '_'),
                                            "value": {"text": "Loading model"},
                                        }
                                    )
                                )
                            
                            load_model(
                                modelData["file"],
                                "scripts/v1-inference.yaml",
                                modelData["device"],
                                modelData["precision"],
                                modelData["optimized"],
                                False
                            )
                            if values["send_progress"]:
                                await websocket.send(
                                    json.dumps(
                                        {
                                            "action": "display_title",
                                            "type": title.lower().replace(' ', '_'),
                                            "value": {"text": "Generating..."},
                                        }
                                    )
                                )
                            
                            # Neural detail workflow
                            
                            # Decode input image
                            init_img = decodeImage(values["images"][0])

                            # Resize image to output dimensions
                            image = init_img.resize((values["width"], values["height"]), resample=Image.Resampling.NEAREST)

                            # Blur filter for detail
                            image_blur = image.filter(ImageFilter.BoxBlur(2))

                            # i2i strength
                            strength = 0.5 + (values["detail"] * 0.05)

                            # Net models, images, and weights in order
                            modelPath, _ = os.path.split(modelData["file"])
                            netPath = os.path.join(modelPath, "CONTROLNET")
                            lecoPath = os.path.join(modelPath, "LECO")
                            loras = values["loras"]
                            loras.append({"file": os.path.join(lecoPath, "detail.leco"), "weight": values["detail"] * -10})
                            controlnets = [{"model_file": os.path.join(netPath, "Tile.safetensors"), "image": image_blur, "weight": 0.5}, {"model_file": os.path.join(netPath, "Composition.safetensors"), "image": image, "weight": 0.5}]

                            for result in neural_inference(
                                modelData["file"],
                                title,
                                controlnets,
                                values["prompt"],
                                values["negative"],
                                values["blip"],
                                False,
                                values["prompt_tuning"],
                                values["width"],
                                values["height"],
                                values["pixel_size"],
                                values["steps"],
                                values["scale"],
                                strength,
                                values["lighting"],
                                values["composition"],
                                values["seed"],
                                values["generations"],
                                values["max_batch_size"],
                                modelData["device"],
                                modelData["precision"],
                                loras,
                                values["send_progress"],
                                values["use_pixelvae"],
                                values["color_map"],
                                values["post_process"],
                                init_img # Pass original, unscaled image
                            ):
                                if values["send_progress"]:
                                    await websocket.send(json.dumps(result))

                            await websocket.send(
                                json.dumps(
                                    {
                                        "action": "returning",
                                        "type": "img2img",
                                        "value": {"images": result["value"]["images"]},
                                    }
                                )
                            )
                        except Exception as e:
                            if "SSLCertVerificationError" in traceback.format_exc():
                                rprint(
                                    f"\n[#ab333d]ERROR: Latent Diffusion Model download failed due to SSL certificate error. Please run 'open /Applications/Python*/Install\ Certificates.command' in a new terminal"
                                )
                            elif (
                                "torch.cuda.OutOfMemoryError" in traceback.format_exc()
                            ):
                                rprint(
                                    f"\n[#ab333d]ERROR: Generation failed due to insufficient GPU resources. If you are running other GPU heavy programs try closing them. Also try lowering the image generation size or maximum batch size"
                                )
                                if modelLM is not None:
                                    rprint(
                                        f"\n[#ab333d]Try disabling LLM enhanced prompts to free up gpu resources"
                                    )
                            else:
                                rprint(f"\n[#ab333d]ERROR:\n{traceback.format_exc()}")
                            play("error.wav")
                            await websocket.send(json.dumps({"action": "error"}))
                    case "resize":
                        try:
                            title = "Neural Resize"

                            # Extract parameters from the message
                            values = message["value"]
                            modelData = values["model"]

                            if values["send_progress"]:
                                await websocket.send(
                                    json.dumps(
                                        {
                                            "action": "display_title",
                                            "type": title.lower().replace(' ', '_'),
                                            "value": {"text": "Loading model"},
                                        }
                                    )
                                )
                            
                            load_model(
                                modelData["file"],
                                "scripts/v1-inference.yaml",
                                modelData["device"],
                                modelData["precision"],
                                modelData["optimized"],
                                False
                            )
                            if values["send_progress"]:
                                await websocket.send(
                                    json.dumps(
                                        {
                                            "action": "display_title",
                                            "type": title.lower().replace(' ', '_'),
                                            "value": {"text": "Generating..."},
                                        }
                                    )
                                )
                            
                            # Neural resize workflow
                            
                            # Decode input image
                            init_img = decodeImage(values["images"][0])

                            # Resize image to output dimensions
                            image = init_img.resize((values["width"], values["height"]), resample=Image.Resampling.BILINEAR)

                            # i2i strength
                            strength = 0.9

                            # Net models, images, and weights in order
                            modelPath, _ = os.path.split(modelData["file"])
                            netPath = os.path.join(modelPath, "CONTROLNET")
                            lecoPath = os.path.join(modelPath, "LECO")
                            loras = values["loras"]
                            loras.append({"file": os.path.join(lecoPath, "detail.leco"), "weight": -40})
                            controlnets = [{"model_file": os.path.join(netPath, "Tile.safetensors"), "image": image, "weight": 0.8}, {"model_file": os.path.join(netPath, "Composition.safetensors"), "image": image, "weight": 0.4}]

                            for result in neural_inference(
                                modelData["file"],
                                title,
                                controlnets,
                                values["prompt"],
                                values["negative"],
                                values["blip"],
                                False,
                                values["prompt_tuning"],
                                values["width"],
                                values["height"],
                                values["pixel_size"],
                                values["steps"],
                                values["scale"],
                                strength,
                                values["lighting"],
                                values["composition"],
                                values["seed"],
                                values["generations"],
                                values["max_batch_size"],
                                modelData["device"],
                                modelData["precision"],
                                values["loras"],
                                values["send_progress"],
                                values["use_pixelvae"],
                                values["color_map"],
                                values["post_process"],
                                init_img # Pass original, unscaled image
                            ):
                                if values["send_progress"]:
                                    await websocket.send(json.dumps(result))

                            await websocket.send(
                                json.dumps(
                                    {
                                        "action": "returning",
                                        "type": "img2img",
                                        "value": {"images": result["value"]["images"]},
                                    }
                                )
                            )
                        except Exception as e:
                            if "SSLCertVerificationError" in traceback.format_exc():
                                rprint(
                                    f"\n[#ab333d]ERROR: Latent Diffusion Model download failed due to SSL certificate error. Please run 'open /Applications/Python*/Install\ Certificates.command' in a new terminal"
                                )
                            elif (
                                "torch.cuda.OutOfMemoryError" in traceback.format_exc()
                            ):
                                rprint(
                                    f"\n[#ab333d]ERROR: Generation failed due to insufficient GPU resources. If you are running other GPU heavy programs try closing them. Also try lowering the image generation size or maximum batch size"
                                )
                                if modelLM is not None:
                                    rprint(
                                        f"\n[#ab333d]Try disabling LLM enhanced prompts to free up gpu resources"
                                    )
                            else:
                                rprint(f"\n[#ab333d]ERROR:\n{traceback.format_exc()}")
                            play("error.wav")
                            await websocket.send(json.dumps({"action": "error"}))
                    case "pixelate":
                        try:
                            title = "Neural Pixelate"

                            # Extract parameters from the message
                            values = message["value"]
                            modelData = values["model"]

                            if values["send_progress"]:
                                await websocket.send(
                                    json.dumps(
                                        {
                                            "action": "display_title",
                                            "type": title.lower().replace(' ', '_'),
                                            "value": {"text": "Loading model"},
                                        }
                                    )
                                )
                            
                            load_model(
                                modelData["file"],
                                "scripts/v1-inference.yaml",
                                modelData["device"],
                                modelData["precision"],
                                modelData["optimized"],
                                False
                            )
                            if values["send_progress"]:
                                await websocket.send(
                                    json.dumps(
                                        {
                                            "action": "display_title",
                                            "type": title.lower().replace(' ', '_'),
                                            "value": {"text": "Generating..."},
                                        }
                                    )
                                )

                            # Decode input image
                            init_img = decodeImage(values["images"][0])

                            # Resize image to output dimensions
                            image = init_img.resize((values["width"], values["height"]), resample=Image.Resampling.BILINEAR)

                            # Blur filter for pixelate
                            image_blur = image.filter(ImageFilter.BoxBlur(2))

                            # i2i strength
                            strength = 0.85

                            # Net models, images, and weights in order
                            modelPath, _ = os.path.split(modelData["file"])
                            netPath = os.path.join(modelPath, "CONTROLNET")
                            controlnets = [{"model_file": os.path.join(netPath, "Tile.safetensors"), "image": image_blur, "weight": 1.0}, {"model_file": os.path.join(netPath, "Composition.safetensors"), "image": image, "weight": 0.6}]

                            for result in neural_inference(
                                modelData["file"],
                                title,
                                controlnets,
                                values["prompt"],
                                values["negative"],
                                values["blip"],
                                False,
                                values["prompt_tuning"],
                                values["width"],
                                values["height"],
                                values["pixel_size"],
                                values["steps"],
                                values["scale"],
                                strength,
                                values["lighting"],
                                values["composition"],
                                values["seed"],
                                values["generations"],
                                values["max_batch_size"],
                                modelData["device"],
                                modelData["precision"],
                                values["loras"],
                                values["send_progress"],
                                values["use_pixelvae"],
                                values["color_map"],
                                values["post_process"],
                                init_img # Pass original, unscaled image
                            ):
                                if values["send_progress"]:
                                    await websocket.send(json.dumps(result))

                            await websocket.send(
                                json.dumps(
                                    {
                                        "action": "returning",
                                        "type": "img2img",
                                        "value": {"images": result["value"]["images"]},
                                    }
                                )
                            )
                        except Exception as e:
                            if "SSLCertVerificationError" in traceback.format_exc():
                                rprint(
                                    f"\n[#ab333d]ERROR: Latent Diffusion Model download failed due to SSL certificate error. Please run 'open /Applications/Python*/Install\ Certificates.command' in a new terminal"
                                )
                            elif (
                                "torch.cuda.OutOfMemoryError" in traceback.format_exc()
                            ):
                                rprint(
                                    f"\n[#ab333d]ERROR: Generation failed due to insufficient GPU resources. If you are running other GPU heavy programs try closing them. Also try lowering the image generation size or maximum batch size"
                                )
                                if modelLM is not None:
                                    rprint(
                                        f"\n[#ab333d]Try disabling LLM enhanced prompts to free up gpu resources"
                                    )
                            else:
                                rprint(f"\n[#ab333d]ERROR:\n{traceback.format_exc()}")
                            play("error.wav")
                            await websocket.send(json.dumps({"action": "error"}))
                    case "txt2img":
                        try:
                            # Extract parameters from the message
                            values = message["value"]
                            modelData = values["model"]

                            if values["send_progress"]:
                                await websocket.send(
                                    json.dumps(
                                        {
                                            "action": "display_title",
                                            "type": "txt2img",
                                            "value": {"text": "Loading model"},
                                        }
                                    )
                                )
                            load_model(
                                modelData["file"],
                                "scripts/v1-inference.yaml",
                                modelData["device"],
                                modelData["precision"],
                                modelData["optimized"],
                            )

                            if values["send_progress"]:
                                await websocket.send(
                                    json.dumps(
                                        {
                                            "action": "display_title",
                                            "type": "txt2img",
                                            "value": {"text": "Generating..."},
                                        }
                                    )
                                )
                            for result in txt2img(
                                values["prompt"],
                                values["negative"],
                                values["translate"],
                                values["prompt_tuning"],
                                values["width"],
                                values["height"],
                                values["pixel_size"],
                                values["enhance_composition"],
                                values["quality"],
                                values["scale"],
                                values["lighting"],
                                values["composition"],
                                values["seed"],
                                values["generations"],
                                values["max_batch_size"],
                                modelData["device"],
                                modelData["precision"],
                                values["loras"],
                                values["tile_x"],
                                values["tile_y"],
                                values["send_progress"],
                                values["use_pixelvae"],
                                values["post_process"],
                            ):
                                if values["send_progress"]:
                                    await websocket.send(json.dumps(result))

                            await websocket.send(
                                json.dumps(
                                    {
                                        "action": "returning",
                                        "type": "txt2img",
                                        "value": {"images": result["value"]["images"]},
                                    }
                                )
                            )
                        except Exception as e:
                            if "SSLCertVerificationError" in traceback.format_exc():
                                rprint(
                                    f"\n[#ab333d]ERROR: Latent Diffusion Model download failed due to SSL certificate error. Please run 'open /Applications/Python*/Install\ Certificates.command' in a new terminal"
                                )
                            elif (
                                "torch.cuda.OutOfMemoryError" in traceback.format_exc()
                            ):
                                rprint(
                                    f"\n[#ab333d]ERROR: Generation failed due to insufficient GPU resources. If you are running other GPU heavy programs try closing them. Also try lowering the image generation size or maximum batch size"
                                )
                                if modelLM is not None:
                                    rprint(
                                        f"\n[#ab333d]Try disabling LLM enhanced prompts to free up gpu resources"
                                    )
                            else:
                                rprint(f"\n[#ab333d]ERROR:\n{traceback.format_exc()}")
                            play("error.wav")
                            await websocket.send(json.dumps({"action": "error"}))
                    case "img2img":
                        try:
                            # Extract parameters from the message
                            values = message["value"]
                            modelData = values["model"]

                            if values["send_progress"]:
                                await websocket.send(
                                    json.dumps(
                                        {
                                            "action": "display_title",
                                            "type": "img2img",
                                            "value": {"text": "Loading model"},
                                        }
                                    )
                                )
                            load_model(
                                modelData["file"],
                                "scripts/v1-inference.yaml",
                                modelData["device"],
                                modelData["precision"],
                                modelData["optimized"],
                            )

                            if values["send_progress"]:
                                await websocket.send(
                                    json.dumps(
                                        {
                                            "action": "display_title",
                                            "type": "img2img",
                                            "value": {"text": "Generating..."},
                                        }
                                    )
                                )
                            for result in img2img(
                                values["prompt"],
                                values["negative"],
                                values["translate"],
                                values["prompt_tuning"],
                                values["width"],
                                values["height"],
                                values["pixel_size"],
                                values["quality"],
                                values["scale"],
                                values["strength"],
                                values["lighting"],
                                values["composition"],
                                values["seed"],
                                values["generations"],
                                values["max_batch_size"],
                                modelData["device"],
                                modelData["precision"],
                                values["loras"],
                                values["images"],
                                values["tile_x"],
                                values["tile_y"],
                                values["send_progress"],
                                values["use_pixelvae"],
                                values["post_process"],
                            ):
                                if values["send_progress"]:
                                    await websocket.send(json.dumps(result))

                            await websocket.send(
                                json.dumps(
                                    {
                                        "action": "returning",
                                        "type": "img2img",
                                        "value": {"images": result["value"]["images"]},
                                    }
                                )
                            )
                        except Exception as e:
                            if "SSLCertVerificationError" in traceback.format_exc():
                                rprint(
                                    f"\n[#ab333d]ERROR: Latent Diffusion Model download failed due to SSL certificate error. Please run 'open /Applications/Python*/Install\ Certificates.command' in a new terminal"
                                )
                            elif (
                                "torch.cuda.OutOfMemoryError" in traceback.format_exc()
                            ):
                                rprint(
                                    f"\n[#ab333d]ERROR: Generation failed due to insufficient GPU resources. If you are running other GPU heavy programs try closing them. Also try lowering the image generation size or maximum batch size. If samples are at 100%, this was caused by the VAE running out of memory, try enabling the Fast Pixel Decoder"
                                )
                                if modelLM is not None:
                                    rprint(
                                        f"\n[#ab333d]Try disabling LLM enhanced prompts to free up gpu resources"
                                    )
                            elif (
                                "Expected batch_size > 0 to be true"
                                in traceback.format_exc()
                            ):
                                rprint(
                                    f"\n[#ab333d]ERROR: Generation failed due to insufficient GPU resources during image encoding. Please lower the maximum batch size, or use a smaller input image"
                                )
                            elif (
                                "cannot reshape tensor of 0 elements"
                                in traceback.format_exc()
                            ):
                                rprint(
                                    f"\n[#ab333d]ERROR: Generation failed due to insufficient GPU resources during image encoding. Please lower the maximum batch size, or use a smaller input image"
                                )
                            else:
                                rprint(f"\n[#ab333d]ERROR:\n{traceback.format_exc()}")
                            play("error.wav")
                            await websocket.send(json.dumps({"action": "error"}))
                    case "txt2pal":
                        try:
                            # Extract parameters from the message
                            values = message["value"]

                            modelData = values["model"]
                            load_model(
                                modelData["file"],
                                "scripts/v1-inference.yaml",
                                modelData["device"],
                                modelData["precision"],
                                modelData["optimized"],
                            )

                            images = paletteGen(
                                values["prompt"],
                                values["colors"],
                                values["seed"],
                                modelData["device"],
                                modelData["precision"],
                            )

                            await websocket.send(
                                json.dumps(
                                    {
                                        "action": "returning",
                                        "type": "txt2pal",
                                        "value": {"images": images},
                                    }
                                )
                            )
                        except Exception as e:
                            if "SSLCertVerificationError" in traceback.format_exc():
                                rprint(
                                    f"\n[#ab333d]ERROR: Latent Diffusion Model download failed due to SSL certificate error. Please run 'open /Applications/Python*/Install\ Certificates.command' in a new terminal"
                                )
                            elif (
                                "torch.cuda.OutOfMemoryError" in traceback.format_exc()
                            ):
                                rprint(
                                    f"\n[#ab333d]ERROR: Generation failed due to insufficient GPU resources. If you are running other GPU heavy programs try closing them. Also try lowering the image generation size or maximum batch size"
                                )
                                if modelLM is not None:
                                    rprint(
                                        f"\n[#ab333d]Try disabling LLM enhanced prompts to free up gpu resources"
                                    )
                            else:
                                rprint(f"\n[#ab333d]ERROR:\n{traceback.format_exc()}")
                            play("error.wav")
                            await websocket.send(json.dumps({"action": "error"}))
                    case "translate":
                        try:
                            # Extract parameters from the message
                            values = message["value"]

                            prompts = prompt2prompt(
                                values["model_folder"],
                                values["prompt"],
                                values["negative"],
                                values["generations"],
                                values["seed"],
                            )
                            await websocket.send(
                                json.dumps(
                                    {
                                        "action": "returning",
                                        "type": "translate",
                                        "value": prompts,
                                    }
                                )
                            )
                        except Exception as e:
                            rprint(f"\n[#ab333d]ERROR:\n{traceback.format_exc()}")
                            play("error.wav")
                            await websocket.send(json.dumps({"action": "error"}))
                    case "benchmark":
                        try:
                            # Extract parameters from the message
                            values = message["value"]

                            modelData = values["model"]
                            load_model(
                                modelData["file"],
                                "scripts/v1-inference.yaml",
                                modelData["device"],
                                modelData["precision"],
                                modelData["optimized"],
                            )

                            prompts = benchmark(
                                modelData["device"],
                                modelData["precision"],
                                values["time_limit"],
                                values["max_test_size"],
                                values["error_range"],
                                values["use_pixelvae"],
                                values["seed"],
                            )

                            await websocket.send(
                                json.dumps(
                                    {
                                        "action": "returning",
                                        "type": "benchmark",
                                        "value": max(32, maxSize - 8),
                                    }
                                )
                            )  # We subtract 8 to leave a little VRAM headroom, so it doesn't OOM if you open a youtube tab T-T
                        except Exception as e:
                            rprint(f"\n[#ab333d]ERROR:\n{traceback.format_exc()}")
                            play("error.wav")
                            await websocket.send(json.dumps({"action": "error"}))
                    case "palettize":
                        try:
                            # Extract parameters from the message
                            values = message["value"]
                            images = palettize(
                                values["images"],
                                values["source"],
                                values["url"],
                                values["palettes"],
                                values["colors"],
                                values["dithering"],
                                values["dither_strength"],
                                values["denoise"],
                                values["smoothness"],
                                values["intensity"],
                            )
                            await websocket.send(
                                json.dumps(
                                    {
                                        "action": "returning",
                                        "type": "palettize",
                                        "value": {"images": images},
                                    }
                                )
                            )
                        except Exception as e:
                            rprint(f"\n[#ab333d]ERROR:\n{traceback.format_exc()}")
                            play("error.wav")
                            await websocket.send(json.dumps({"action": "error"}))
                    case "rembg":
                        try:
                            # Extract parameters from the message
                            values = message["value"]
                            images = rembg(values["images"], values["model_folder"])
                            await websocket.send(
                                json.dumps(
                                    {
                                        "action": "returning",
                                        "type": "rembg",
                                        "value": {"images": images},
                                    }
                                )
                            )
                        except Exception as e:
                            rprint(f"\n[#ab333d]ERROR:\n{traceback.format_exc()}")
                            play("error.wav")
                            await websocket.send(json.dumps({"action": "error"}))
                    case "pixelDetect":
                        try:
                            # Extract parameters from the message
                            values = message["value"]
                            images = pixelDetectVerbose(values["images"])
                            await websocket.send(
                                json.dumps(
                                    {
                                        "action": "returning",
                                        "type": "pixelDetect",
                                        "value": {"images": images},
                                    }
                                )
                            )
                        except Exception as e:
                            rprint(f"\n[#ab333d]ERROR:\n{traceback.format_exc()}")
                            play("error.wav")
                            await websocket.send(json.dumps({"action": "error"}))
                    case "kcentroid":
                        try:
                            # Extract parameters from the message
                            values = message["value"]
                            images = kCentroidVerbose(
                                values["images"],
                                values["width"],
                                values["height"],
                                values["centroids"],
                            )
                            await websocket.send(
                                json.dumps(
                                    {
                                        "action": "returning",
                                        "type": "kcentroid",
                                        "value": {"images": images},
                                    }
                                )
                            )
                        except Exception as e:
                            rprint(f"\n[#ab333d]ERROR:\n{traceback.format_exc()}")
                            play("error.wav")
                            await websocket.send(json.dumps({"action": "error"}))
                    case "connected":
                        global sounds
                        try:
                            background, sounds, extensionVersion = (
                                message["value"]["background"],
                                message["value"]["play_sound"],
                                message["value"]["version"],
                            )
                            rd = gw.getWindowsWithTitle(
                                "Retro Diffusion Image Generator"
                            )[0]
                            if not background:
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
                            await websocket.send(json.dumps({"action": "connected"}))
                        else:
                            rprint(
                                f"\n[#ab333d]The current client is on a version that is incompatible with the image generator version. Please update the extension."
                            )
                    case "recieved":
                        if not background:
                            try:
                                rd = gw.getWindowsWithTitle(
                                    "Retro Diffusion Image Generator"
                                )[0]
                                if gw.getActiveWindow() is not None:
                                    if (
                                        gw.getActiveWindow().title
                                        == "Retro Diffusion Image Generator"
                                    ):
                                        # Minimize the window
                                        rd.minimize()
                            except:
                                pass
                        await websocket.send(json.dumps({"action": "free_websocket"}))
                        clearCache()
                    case "shutdown":
                        rprint("[#ab333d]Shutting down...")
                        global running
                        global timeout
                        running = False
                        await websocket.close()
                        asyncio.get_event_loop().call_soon_threadsafe(
                            asyncio.get_event_loop().stop
                        )
            except:
                pass
    except Exception as e:
        if "asyncio.exceptions.IncompleteReadError" in traceback.format_exc():
            rprint(f"\n[#ab333d]Bytes read error (resolved automatically)")
        else:
            if "PayloadTooBig" in traceback.format_exc() or "message too big" in traceback.format_exc():
                rprint(f"\n[#ab333d]ERROR:\n{traceback.format_exc()}\n\n\n[#ab333d]Websockets received a message that was too large, unless accompanied by other errors this is safe to ignore.")
            else:
                rprint(f"\n[#ab333d]ERROR:\n{traceback.format_exc()}")
            play("error.wav")


if system == "Windows":
    os.system("title Retro Diffusion Image Generator")
elif system == "Darwin":
    os.system("printf '\\033]0;Retro Diffusion Image Generator\\007'")

rprint("\n" + climage(Image.open("logo.png"), "centered") + "\n\n")

rprint("[#48a971]Starting Image Generator...")

start_server = serve(server, "localhost", 8765, max_size=100 * 1024 * 1024)

rprint("[#c4f129]Connected")

timeout = 1

# Run the server until it is completed
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()

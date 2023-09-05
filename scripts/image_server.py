# Import core libraries
import os, re, time, asyncio, ctypes
from scripts.generators.img2img import img2img
from scripts.generators.paletteGen import paletteGen
from scripts.generators.txt2img import txt2img
from scripts.retro_diffusion import rd
from scripts.util.audio import play
from scripts.util.models import prepare_model_for_inference
from scripts.util.palettize import palettize
from scripts.util.parse_bool import parse_bool
from scripts.util.pixel_detect import pixelDetectVerbose
from scripts.util.rembg import rembg
from scripts.util.search_string import searchString
from scripts.util.climage import climage
from scripts.util.kCentroid import kCentroidVerbose
from sdkit.models import load_model as sdkit_load_model
import torch

# Import logging libraries
import traceback, warnings

# Import websocket tools
from websockets import serve, connect

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    try:
        import pygetwindow as gw
    except:
        rd.logger(
            f"[#ab333d]Pygetwindow could not be loaded. This will limit some cosmetic functionality."
        )
from colorama import just_fix_windows_console

# Fix windows console for color codes
just_fix_windows_console()

# Patch existing console to remove interactivity
kernel32 = ctypes.windll.kernel32
kernel32.SetConsoleMode(kernel32.GetStdHandle(-10), 128)

expectedVersion = "7.5.0"

global timeout
global loaded
loaded = ""

async def server(websocket):
    background = False

    async for message in websocket:
        if re.search(r"txt2img.+", message):
            await websocket.send("running txt2img")
            try:
                # Extract parameters from the message
                (
                    loraPath,
                    loraFiles,
                    loraWeights,
                    device,
                    precision,
                    pixelSize,
                    prompt,
                    negative,
                    w,
                    h,
                    ddim_steps,
                    scale,
                    seed,
                    n_iter,
                    tilingX,
                    tilingY,
                    pixelvae,
                    post,
                ) = searchString(
                    message,
                    "dlorapath",
                    "dlorafiles",
                    "dloraweights",
                    "ddevice",
                    "dprecision",
                    "dpixelsize",
                    "dprompt",
                    "dnegative",
                    "dwidth",
                    "dheight",
                    "dstep",
                    "dscale",
                    "dseed",
                    "diter",
                    "dtilingx",
                    "dtilingy",
                    "dpixelvae",
                    "dpalettize",
                    "end",
                )
                
                loraFiles = loraFiles.split("|")
                loraWeights = [int(x) for x in loraWeights.split("|")]
                
                # ControlNet test
                from PIL import Image
                control_image = Image.open("cheems.png")
                import cv2
                import numpy as np
                canny_image = cv2.Canny(np.array(control_image), 100, 200)
                # convert back to PIL
                canny_image = Image.fromarray(canny_image)
                canny_image.show()
                
                txt2img(
                    loraPath,
                    loraFiles,
                    loraWeights,
                    device,
                    precision,
                    int(pixelSize),
                    prompt,
                    negative,
                    int(w),
                    int(h),
                    int(ddim_steps),
                    float(scale),
                    int(seed),
                    int(n_iter),
                    tilingX,
                    tilingY,
                    parse_bool(pixelvae),
                    parse_bool(post),
                    canny_image
                )
                
                await websocket.send("returning txt2img")
            except Exception as e:
                rd.logger(f"\n[#ab333d]ERROR:\n{traceback.format_exc()}")
                play("error.wav")
                await websocket.send("returning error")

        elif re.search(r"txt2pal.+", message):
            await websocket.send("running txt2pal")
            try:
                # Extract parameters from the message
                device, precision, prompt, seed, colors = searchString(
                    message,
                    "ddevice",
                    "dprecision",
                    "dprompt",
                    "dseed",
                    "dcolors",
                    "end",
                )
                paletteGen(int(colors), device, precision, prompt, int(seed))
                await websocket.send("returning txt2pal")
            except Exception as e:
                rd.logger(f"\n[#ab333d]ERROR:\n{traceback.format_exc()}")
                play("error.wav")
                await websocket.send("returning error")

        elif re.search(r"img2img.+", message):
            await websocket.send("running img2img")
            try:
                # Extract parameters from the message
                (
                    loraPath,
                    loraFiles,
                    loraWeights,
                    device,
                    precision,
                    pixelSize,
                    prompt,
                    negative,
                    w,
                    h,
                    ddim_steps,
                    scale,
                    strength,
                    seed,
                    n_iter,
                    tilingX,
                    tilingY,
                    pixelvae,
                    post,
                ) = searchString(
                    message,
                    "dlorapath",
                    "dlorafiles",
                    "dloraweights",
                    "ddevice",
                    "dprecision",
                    "dpixelsize",
                    "dprompt",
                    "dnegative",
                    "dwidth",
                    "dheight",
                    "dstep",
                    "dscale",
                    "dstrength",
                    "dseed",
                    "diter",
                    "dtilingx",
                    "dtilingy",
                    "dpixelvae",
                    "dpalettize",
                    "end",
                )
                
                loraFiles = loraFiles.split("|")
                loraWeights = [int(x) for x in loraWeights.split("|")]
                
                img2img(
                    loraPath,
                    loraFiles,
                    loraWeights,
                    device,
                    precision,
                    int(pixelSize),
                    prompt,
                    negative,
                    int(w),
                    int(h),
                    int(ddim_steps),
                    float(scale),
                    float(strength) / 100,
                    int(seed),
                    int(n_iter),
                    tilingX,
                    tilingY,
                    parse_bool(pixelvae),
                    parse_bool(post)
                )
                
                await websocket.send("returning img2img")
            except Exception as e:
                rd.logger(f"\n[#ab333d]ERROR:\n{traceback.format_exc()}")
                play("error.wav")
                await websocket.send("returning error")

        elif re.search(r"palettize.+", message):
            await websocket.send("running palettize")
            try:
                # Extract parameters from the message
                (
                    numFiles,
                    source,
                    colors,
                    bestPaletteFolder,
                    paletteFile,
                    paletteURL,
                    dithering,
                    strength,
                    denoise,
                    smoothness,
                    intensity,
                ) = searchString(
                    message,
                    "dnumfiles",
                    "dsource",
                    "dcolors",
                    "dbestpalettefolder",
                    "dpalettefile",
                    "dpaletteURL",
                    "ddithering",
                    "dstrength",
                    "ddenoise",
                    "dsmoothness",
                    "dintensity",
                    "end",
                )
                palettize(
                    int(numFiles),
                    source,
                    int(colors),
                    bestPaletteFolder,
                    paletteFile,
                    paletteURL,
                    int(dithering),
                    int(strength),
                    denoise,
                    int(smoothness),
                    int(intensity),
                )
                await websocket.send("returning palettize")
            except Exception as e:
                rd.logger(f"\n[#ab333d]ERROR:\n{traceback.format_exc()}")
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
                rd.logger(f"\n[#ab333d]ERROR:\n{traceback.format_exc()}")
                play("error.wav")
                await websocket.send("returning error")

        elif re.search(r"pixelDetect.+", message):
            await websocket.send("running pixelDetect")
            try:
                pixelDetectVerbose()
                await websocket.send("returning pixelDetect")
            except Exception as e:
                rd.logger(f"\n[#ab333d]ERROR:\n{traceback.format_exc()}")
                play("error.wav")
                await websocket.send("returning error")

        elif re.search(r"kcentroid.+", message):
            await websocket.send("running kcentroid")
            try:
                # Extract parameters from the message
                width, height, centroids = searchString(
                    message, "dwidth", "dheight", "dcentroids", "end"
                )
                kCentroidVerbose(int(width), int(height), int(centroids))
                await websocket.send("returning kcentroid")
            except Exception as e:
                rd.logger(f"\n[#ab333d]ERROR:\n{traceback.format_exc()}")
                play("error.wav")
                await websocket.send("returning error")

        elif re.search(r"load.+", message):
            timer = time.time()
            await websocket.send("loading model")
            global loaded
            if loaded != message:
                try:
                    # Extract parameters from the message
                    device, optimized, precision, path, model = searchString(
                        message,
                        "ddevice",
                        "doptimized",
                        "dprecision",
                        "dpath",
                        "dmodel",
                        "end",
                    )
                    
                    # Set VRAM optimizations
                    # rd.context.vram_usage_level = "low"

                    # Load sdkit model
                    # model = "sd_xl_base_1.0_0.9vae.safetensors" # sdxl test
                    new_path = prepare_model_for_inference(f"{path}{model}")
                    rd.context.model_paths["stable-diffusion"] = new_path
                    sdkit_load_model(rd.context, "stable-diffusion")
                    
                    # Testing controlnet
                    rd.context.model_paths["controlnet"] = "models/controlnet/control_v11p_sd15_lineart.pth"
                    sdkit_load_model(rd.context, "controlnet")

                    play("iteration.wav")
                    # rd.logger(f"[#c4f129]Loaded model to [#48a971]{model.cdevice}[#c4f129] at [#48a971]{precision} precision[#c4f129] in [#48a971]{round(time.time()-timer, 2)} [#c4f129]seconds")
                    loaded = message
                except Exception as e:
                    rd.logger(f"\n[#ab333d]ERROR:\n{traceback.format_exc()}")
                    play("error.wav")
                    await websocket.send("returning error")

            await websocket.send("loaded model")

        elif re.search(r"connected.+", message):
            try:
                background, sounds, extensionVersion = searchString(
                    message, "dbackground", "dsound", "dversion", "end"
                )
                rd.sounds = parse_bool(sounds)
                rd_window = gw.getWindowsWithTitle("Retro Diffusion Image Generator")[0]
                if background == "false":
                    try:
                        # Restore and activate the window
                        rd_window.restore()
                        rd_window.activate()
                    except:
                        pass
                else:
                    try:
                        # Minimize the window
                        rd_window.minimize()
                    except:
                        pass
            except:
                pass

            if extensionVersion == expectedVersion:
                play("click.wav")
                await websocket.send("connected")
            else:
                rd.logger(
                    f"\n[#ab333d]The current client is on a version that is incompatible with the image generator version. Please update the extension."
                )

        elif message == "no model":
            await websocket.send("loaded model")
        elif message == "recieved":
            if background == "false":
                try:
                    rd_window = gw.getWindowsWithTitle("Retro Diffusion Image Generator")[0]
                    if gw.getActiveWindow() is not None:
                        if (
                            gw.getActiveWindow().title
                            == "Retro Diffusion Image Generator"
                        ):
                            # Minimize the window
                            rd_window.minimize()
                except:
                    pass
            await websocket.send("free")
            torch.cuda.empty_cache()
        elif message == "shutdown":
            rd.logger("[#ab333d]Shutting down...")
            global running
            global timeout
            running = False
            await websocket.close()
            asyncio.get_event_loop().call_soon_threadsafe(asyncio.get_event_loop().stop)


async def connectSend(uri, message):
    async with connect(uri) as websocket:
        # Send a message over the WebSocket connection
        await websocket.send(message)


os.system("title Retro Diffusion Image Generator")

rd.logger("\n" + climage("logo.png", "centered") + "\n\n")

rd.logger("[#48a971]Starting Image Generator...")

start_server = serve(server, "localhost", 8765)

rd.logger("[#c4f129]Connected")

timeout = 1

# Run the server until it is completed
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()

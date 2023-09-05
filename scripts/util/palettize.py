from io import BytesIO
import os
from PIL import Image
import numpy as np
import requests
from scripts.retro_diffusion import rd
import time
from scripts.util.adjust_gamma import adjust_gamma
from scripts.util.audio import play
import hitherdither

from scripts.util.clbar import clbar
from scripts.util.determine_best_k_verbose import determine_best_k_verbose
from scripts.util.kDenoise import kDenoise

def palettize(
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
):
    # Check if a palette URL is provided and try to download the palette image
    if source == "URL":
        try:
            paletteFile = BytesIO(requests.get(paletteURL).content)
            testImg = Image.open(paletteFile).convert("RGB")
        except:
            rd.logger(
                f"\n[#ab333d]ERROR: URL {paletteURL} cannot be reached or is not an image\nReverting to Adaptive palette"
            )
            paletteFile = ""

    timer = time.time()

    # Create a list to store file paths
    files = []
    for n in range(numFiles):
        files.append(f"temp/input{n+1}.png")

    # Determine the number of colors based on the palette or user input
    if paletteFile != "":
        palImg = Image.open(paletteFile).convert("RGB")
        numColors = len(palImg.getcolors(16777216))
    else:
        numColors = colors

    # Create the string for conversion message
    string = (
        f"\n[#48a971]Converting output[white] to [#48a971]{numColors}[white] colors"
    )

    # Add dithering information if strength and dithering are greater than 0
    if strength > 0 and dithering > 0:
        string = f"{string} with order [#48a971]{dithering}[white] dithering"

    if source == "Automatic":
        string = f"\n[#48a971]Converting output[white] to best number of colors"
    elif source == "Best Palette":
        string = f"\n[#48a971]Converting output[white] to best color palette"

    # Print the conversion message
    rd.logger(string)

    palFiles = []
    # Process each file in the list
    for file in clbar(
        files,
        name="Processed",
        position="last",
        unit="image",
        prefixwidth=12,
        suffixwidth=28,
    ):
        img = Image.open(file).convert("RGB")

        # Apply denoising if enabled
        if denoise == "true":
            img = kDenoise(img, smoothness, intensity)

        # Calculate the threshold for dithering
        threshold = 4 * strength

        if source == "Automatic":
            numColors = determine_best_k_verbose(img, 64)

        # Check if a palette file is provided
        if (paletteFile != "" and os.path.isfile(file)) or source == "Best Palette":
            # Open the palette image and calculate the number of colors
            if source == "Best Palette":
                palImg, palFile = determine_best_palette_verbose(img, bestPaletteFolder)
                palFiles.append(str(palFile))
            else:
                palImg = Image.open(paletteFile).convert("RGB")

            numColors = len(palImg.getcolors(16777216))

            if strength > 0 and dithering > 0:
                for _ in clbar(
                    [img],
                    name="Palettizing",
                    position="first",
                    prefixwidth=12,
                    suffixwidth=28,
                ):
                    # Adjust the image gamma
                    img = adjust_gamma(img, 1.0 - (0.02 * strength))

                    # Extract palette colors
                    palette = [x[1] for x in palImg.getcolors(16777216)]

                    # Perform ordered dithering using Bayer matrix
                    palette = hitherdither.palette.Palette(palette)
                    img_indexed = hitherdither.ordered.bayer.bayer_dithering(
                        img, palette, [threshold, threshold, threshold], order=dithering
                    ).convert("RGB")
            else:
                # Extract palette colors
                palette = np.concatenate(
                    [x[1] for x in palImg.getcolors(16777216)]
                ).tolist()

                # Create a new palette image
                palImg = Image.new("P", (256, 1))
                palImg.putpalette(palette)

                # Perform quantization without dithering
                for _ in clbar(
                    [img],
                    name="Palettizing",
                    position="first",
                    prefixwidth=12,
                    suffixwidth=28,
                ):
                    img_indexed = img.quantize(
                        method=1, kmeans=numColors, palette=palImg, dither=0
                    ).convert("RGB")

        elif numColors > 0 and os.path.isfile(file):
            if strength > 0 and dithering > 0:
                # Perform quantization with ordered dithering
                for _ in clbar(
                    [img],
                    name="Palettizing",
                    position="first",
                    prefixwidth=12,
                    suffixwidth=28,
                ):
                    img_indexed = img.quantize(
                        colors=numColors, method=1, kmeans=numColors, dither=0
                    ).convert("RGB")

                    # Adjust the image gamma
                    img = adjust_gamma(img, 1.0 - (0.03 * strength))

                    # Extract palette colors
                    palette = [x[1] for x in img_indexed.getcolors(16777216)]

                    # Perform ordered dithering using Bayer matrix
                    palette = hitherdither.palette.Palette(palette)
                    img_indexed = hitherdither.ordered.bayer.bayer_dithering(
                        img, palette, [threshold, threshold, threshold], order=dithering
                    ).convert("RGB")

            else:
                # Perform quantization without dithering
                for _ in clbar(
                    [img],
                    name="Palettizing",
                    position="first",
                    prefixwidth=12,
                    suffixwidth=28,
                ):
                    img_indexed = img.quantize(
                        colors=numColors, method=1, kmeans=numColors, dither=0
                    ).convert("RGB")

        img_indexed.save(file)

        if file != files[-1]:
            play("iteration.wav")
        else:
            play("batch.wav")

    rd.logger(
        f"[#c4f129]Palettized [#48a971]{len(files)}[#c4f129] images in [#48a971]{round(time.time()-timer, 2)}[#c4f129] seconds"
    )
    if source == "Best Palette":
        rd.logger(f"[#c4f129]Palettes used: [#494b9b]{', '.join(palFiles)}")
        
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
    for palImg in clbar(
        paletteImages,
        name="Searching",
        position="first",
        prefixwidth=12,
        suffixwidth=28,
    ):
        try:
            palImg = Image.open(f"{paletteFolder}/{palImg}").convert("RGB")
        except:
            continue
        palette = []

        # Extract palette colors
        palColors = palImg.getcolors(16777216)
        numColors = len(palColors)
        palette = np.concatenate([x[1] for x in palColors]).tolist()

        # Create a new palette image
        palImg = Image.new("P", (256, 1))
        palImg.putpalette(palette)

        quantized_image = image.quantize(
            method=1, kmeans=numColors, palette=palImg, dither=0
        )
        centroids = np.array(quantized_image.getpalette()[: numColors * 3]).reshape(
            -1, 3
        )

        # Calculate distortions
        distances = np.linalg.norm(pixel_indices[:, np.newaxis] - centroids, axis=2)
        min_distances = np.min(distances, axis=1)
        distortions.append(np.sum(min_distances**2))

    # Find the best match
    best_match_index = np.argmin(distortions)
    best_palette = Image.open(
        f"{paletteFolder}/{paletteImages[best_match_index]}"
    ).convert("RGB")

    return best_palette, paletteImages[best_match_index]

def palettizeOutput(numFiles):
    # Create a list to store file paths
    files = []
    for n in range(numFiles):
        files.append(f"temp/temp{n+1}.png")

    # Process the image using pixelDetect and save the result
    for file in clbar(
        files,
        name="Processed",
        position="last",
        unit="image",
        prefixwidth=12,
        suffixwidth=28,
    ):
        img = Image.open(file).convert("RGB")

        numColors = determine_best_k_verbose(img, 64)

        for _ in clbar(
            [img], name="Palettizing", position="first", prefixwidth=12, suffixwidth=28
        ):
            img_indexed = img.quantize(
                colors=numColors, method=1, kmeans=numColors, dither=0
            ).convert("RGB")

            img_indexed.save(file)
        if file != files[-1]:
            play("iteration.wav")
        else:
            play("batch.wav")
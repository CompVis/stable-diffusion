import os
from PIL import Image
import numpy as np
import scipy
from scripts.util.audio import play
from scripts.util.clbar import clbar
from scripts.util.determine_best_k_verbose import determine_best_k_verbose
from scripts.util.kCentroid import kCentroid
from scripts.retro_diffusion import rd

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
    return kCentroid(
        image,
        round(image.width / np.median(hspacing)),
        round(image.height / np.median(vspacing)),
        2,
    )
    
def pixelDetectVerbose():
    # Check if input file exists and open it
    assert os.path.isfile("temp/input.png")
    init_img = Image.open("temp/input.png")

    rd.logger(f"\n[#48a971]Finding pixel ratio for current cel")

    # Process the image using pixelDetect and save the result
    for _ in clbar(
        range(1),
        name="Processed",
        position="last",
        unit="image",
        prefixwidth=12,
        suffixwidth=28,
    ):
        downscale = pixelDetect(init_img)

        numColors = determine_best_k_verbose(downscale, 64)

        for _ in clbar(
            [downscale],
            name="Palettizing",
            position="first",
            prefixwidth=12,
            suffixwidth=28,
        ):
            img_indexed = downscale.quantize(
                colors=numColors, method=1, kmeans=numColors, dither=0
            ).convert("RGB")

        img_indexed.save("temp/temp.png")
    play("batch.wav")
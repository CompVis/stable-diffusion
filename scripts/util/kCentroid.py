import os
from PIL import Image
import numpy as np
from itertools import product
from scripts.util.audio import play

from scripts.util.clbar import clbar
from scripts.retro_diffusion import rd

def kCentroid(image, width, height, centroids):
    image = image.convert("RGB")

    # Create an empty array for the downscaled image
    downscaled = np.zeros((height, width, 3), dtype=np.uint8)

    # Calculate the scaling factors
    wFactor = image.width / width
    hFactor = image.height / height

    # Iterate over each tile in the downscaled image
    for x, y in product(range(width), range(height)):
        # Crop the tile from the original image
        tile = image.crop(
            (x * wFactor, y * hFactor, (x * wFactor) + wFactor, (y * hFactor) + hFactor)
        )

        # Quantize the colors of the tile using k-means clustering
        tile = tile.quantize(colors=centroids, method=1, kmeans=centroids).convert(
            "RGB"
        )

        # Get the color counts and find the most common color
        color_counts = tile.getcolors()
        most_common_color = max(color_counts, key=lambda x: x[0])[1]

        # Assign the most common color to the corresponding pixel in the downscaled image
        downscaled[y, x, :] = most_common_color

    return Image.fromarray(downscaled, mode="RGB")

def kCentroidVerbose(width, height, centroids):
    # Check if the input file exists and open it
    assert os.path.isfile("temp/input.png")
    init_img = Image.open("temp/input.png")

    rd.logger(
        f"\n[#48a971]K-Centroid downscaling[white] from [#48a971]{init_img.width}[white]x[#48a971]{init_img.height}[white] to [#48a971]{width}[white]x[#48a971]{height}[white] with [#48a971]{centroids}[white] centroids"
    )

    # Perform k-centroid downscaling and save the image
    for _ in clbar(
        range(1), name="Processed", unit="image", prefixwidth=12, suffixwidth=28
    ):
        kCentroid(init_img, int(width), int(height), int(centroids)).save(
            "temp/temp.png"
        )
    play("batch.wav")
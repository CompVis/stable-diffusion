from PIL import Image
import numpy as np
from itertools import product

def kDenoise(image, smoothing, strength):
    image = image.convert("RGB")

    # Create an array to store the denoised image
    denoised = np.zeros((image.height, image.width, 3), dtype=np.uint8)

    # Iterate over each pixel
    for x, y in product(range(image.width), range(image.height)):
        # Crop the image to a 3x3 tile around the current pixel
        tile = image.crop(
            (x - 1, y - 1, min(x + 2, image.width), min(y + 2, image.height))
        )

        # Calculate the number of centroids based on the tile size and strength
        centroids = max(
            2,
            min(
                round((tile.width * tile.height) * (1 / strength)),
                (tile.width * tile.height),
            ),
        )

        # Quantize the tile to the specified number of centroids
        tile = tile.quantize(colors=centroids, method=1, kmeans=centroids).convert(
            "RGB"
        )

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
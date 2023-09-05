import numpy as np
from scripts.util.clbar import clbar

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
    for k in clbar(
        range(4, round(max_k / 8) + 2),
        name="Finding K",
        position="first",
        prefixwidth=12,
        suffixwidth=28,
    ):
        for n in range(round(max_k / k)):
            count += 1
            quantized_image = image.quantize(
                colors=count, method=2, kmeans=count, dither=0
            )
            centroids = np.array(quantized_image.getpalette()[: count * 3]).reshape(
                -1, 3
            )

            # Calculate distortions
            distances = np.linalg.norm(pixel_indices[:, np.newaxis] - centroids, axis=2)
            min_distances = np.min(distances, axis=1)
            distortions.append(np.sum(min_distances**2))

    # Calculate the rate of change of distortions
    rate_of_change = np.diff(distortions) / np.array(distortions[:-1])

    # Find the elbow point (best k value)
    if len(rate_of_change) == 0:
        best_k = 1
    else:
        elbow_index = np.argmax(rate_of_change)
        best_k = elbow_index + 2

    return best_k
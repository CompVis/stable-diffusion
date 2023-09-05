from PIL import Image
from scripts.util.kCentroid import kCentroid

def image_postprocessing(image: Image, width: int, height: int, pixelSize: int, is_1bit: bool):
    if image.width > int(
        width / pixelSize
    ) and image.height > int(height / pixelSize):
        # Resize the image if pixel is true
        image = kCentroid(
            image, int(width / pixelSize), int(height / pixelSize), 2
        )
    elif image.width < int(
        width / pixelSize
    ) and image.height < int(height / pixelSize):
        image = image.resize(
            (width, height), resample=Image.Resampling.NEAREST
        )

    if is_1bit:
        image = image.quantize(
            colors=4, method=1, kmeans=4, dither=0
        ).convert("RGB")
        image = image.quantize(
            colors=2, method=1, kmeans=2, dither=0
        ).convert("RGB")
        pixels = list(image.getdata())
        darkest, brightest = min(pixels), max(pixels)
        new_pixels = [
            0 if pixel == darkest else 255 if pixel == brightest else pixel
            for pixel in pixels
        ]
        new_image = Image.new("L", image.size)
        new_image.putdata(new_pixels)
        image = new_image.convert("RGB")
    
    return image
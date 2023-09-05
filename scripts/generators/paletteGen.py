import math
from PIL import Image
from scripts.generators.txt2img import txt2img
from scripts.util.kCentroid import kCentroid
from scripts.retro_diffusion import rd

def paletteGen(colors, device, precision, prompt, seed):
    pass
    # Calculate the base for palette generation
    base = 2 ** round(math.log2(colors))

    # Calculate the width of the image based on the base and number of colors
    width = 512 + ((512 / base) * (colors - base))

    # Generate text-to-image conversion with specified parameters
    txt2img(
        None,
        [],
        [],
        device,
        precision,
        1,
        prompt,
        "",
        int(width),
        512,
        20,
        7.0,
        int(seed),
        1,
        "false",
        "false",
        False,
        False,
    )

    # Open the generated image
    image = Image.open("temp/temp1.png").convert("RGB")

    # Perform k-centroid downscaling on the image
    image = kCentroid(image, int(image.width / (512 / base)), 1, 2)

    # Iterate over the pixels in the image and set corresponding palette colors
    palette = Image.new("P", (colors, 1))
    for x in range(image.width):
        for y in range(image.height):
            r, g, b = image.getpixel((x, y))

            palette.putpixel((x, y), (r, g, b))

    palette.save("temp/temp1.png")
    rd.logger(
        f"[#c4f129]Image converted to color palette with [#48a971]{colors}[#c4f129] colors"
    )
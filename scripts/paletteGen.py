from PIL import Image
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--file", required = True, help = "Path to image file")
ap.add_argument("-s", "--size", required = True, type = int, default=0, help = "# of colors")
args = vars(ap.parse_args())

if os.path.isfile(args["file"]) and args["size"] > 0:

    image = Image.open(args["file"])

    image = image.convert('RGB')

    image = image.resize((int(image.width/(512/args["size"])), int(image.height/512)), resample=3)

    colors = []

    palette = Image.new('P', (args["size"], 1))

    for x in range(image.width):
        for y in range(image.height):
            r, g, b = image.getpixel((x, y))

            palette.putpixel((x, y), (r, g, b))

    print(f'{args["file"]} converted to palette image with {args["size"]} colors.')

    palette.save(args["file"])

elif args["size"] <= 0:
    print("Size must be greater than 0.")

else:
    print(f'{args["file"]} does not exist.')
from PIL import Image
import hitherdither
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--file", required = True, help = "Path to image file")
ap.add_argument("-c", "--clusters", required = False, type = int, default=0, help = "# of clusters")
ap.add_argument("-p", "--palette", required = False, default='None', help = "Path to palette file")
ap.add_argument("-b", "--bayer", required = False, type = int, default=0, help = "Bayer matrix order, must be powers of 2")
args = vars(ap.parse_args())

img = Image.open(args["file"]).convert('RGB')
palette = []

if args["bayer"] == 1:
    args["bayer"] = 0
elif (img.width or img.height) >= 256:
    print("Image size is above 256x256 pixels. Please be patient.")

if args["palette"] != 'None' and os.path.isfile(args["file"]):

    palImg = Image.open(args["palette"]).convert('RGB')
    numColors = len(palImg.getcolors(16777216))

    if args["bayer"] > 0:
        if numColors <= 64:
            print(f'Palettizing output to {args["palette"]} colors with order {args["bayer"]} dithering...')

            for i in palImg.getcolors(16777216): 
                palette.append(i[1])
            palette = hitherdither.palette.Palette(palette)
            img_indexed = hitherdither.ordered.yliluoma.yliluomas_1_ordered_dithering(img, palette, order=args["bayer"]).convert('RGB')
        else:
            print('Palette too large, max colors for dithering is 64.')
            print(f'Palettizing output to {args["palette"]} colors...')

            for i in palImg.getcolors(16777216):
                palette.append(i[1][0])
                palette.append(i[1][1])
                palette.append(i[1][2])
            palImg = Image.new('P', (256, 1))
            palImg.putpalette(palette)
            img_indexed = img.quantize(method=1, kmeans=numColors, palette=palImg, dither=0).convert('RGB')
    else:
        print(f'Palettizing output to {args["palette"]} colors...')

        for i in palImg.getcolors(16777216):
            palette.append(i[1][0])
            palette.append(i[1][1])
            palette.append(i[1][2])
        palImg = Image.new('P', (256, 1))
        palImg.putpalette(palette)
        img_indexed = img.quantize(method=1, kmeans=numColors, palette=palImg, dither=0).convert('RGB')

elif args["clusters"] > 0 and os.path.isfile(args["file"]):

    if args["bayer"] > 0:
        if args["clusters"] <= 64:
            print(f'Palettizing output to {args["clusters"]} colors with order {args["bayer"]} dithering...')

            img_indexed = img.quantize(colors=args["clusters"], method=1, kmeans=args["clusters"], dither=0).convert('RGB')

            for i in img_indexed.convert("RGB").getcolors(16777216): 
                palette.append(i[1])
            palette = hitherdither.palette.Palette(palette)
            img_indexed = hitherdither.ordered.yliluoma.yliluomas_1_ordered_dithering(img, palette, order=args["bayer"]).convert('RGB')
        else:
            print('Palette too large, max colors for dithering is 64.')
            img_indexed = img.quantize(colors=args["clusters"], method=1, kmeans=args["clusters"], dither=0).convert('RGB')
    else:
        print(f'Palettizing output to {args["clusters"]} colors...')
        img_indexed = img.quantize(colors=args["clusters"], method=1, kmeans=args["clusters"], dither=0).convert('RGB')

img_indexed.save(args["file"])
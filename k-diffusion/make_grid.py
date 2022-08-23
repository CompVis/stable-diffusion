#!/usr/bin/env python3

"""Assembles images into a grid."""

import argparse
import math
import sys

from PIL import Image


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('images', type=str, nargs='+', metavar='image',
                   help='the input images')
    p.add_argument('--output', '-o', type=str, default='out.png',
                   help='the output image')
    p.add_argument('--nrow', type=int,
                   help='the number of images per row')
    args = p.parse_args()

    images = [Image.open(image) for image in args.images]
    mode = images[0].mode
    size = images[0].size
    for image, name in zip(images, args.images):
        if image.mode != mode:
            print(f'Error: Image {name} had mode {image.mode}, expected {mode}', file=sys.stderr)
            sys.exit(1)
        if image.size != size:
            print(f'Error: Image {name} had size {image.size}, expected {size}', file=sys.stderr)
            sys.exit(1)

    n = len(images)
    x = args.nrow if args.nrow else math.ceil(n**0.5)
    y = math.ceil(n / x)

    output = Image.new(mode, (size[0] * x, size[1] * y))
    for i, image in enumerate(images):
        cur_x, cur_y = i % x, i // x
        output.paste(image, (size[0] * cur_x, size[1] * cur_y))

    output.save(args.output)


if __name__ == '__main__':
    main()

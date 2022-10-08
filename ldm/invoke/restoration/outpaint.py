import warnings
import math
from PIL import Image, ImageFilter

class Outpaint(object):
    def __init__(self, image, generate):
        self.image     = image
        self.generate  = generate

    def process(self, opt, old_opt, image_callback = None, prefix = None):
        image = self._create_outpaint_image(self.image, opt.out_direction)

        seed   = old_opt.seed
        prompt = old_opt.prompt

        def wrapped_callback(img,seed,**kwargs):
            image_callback(img,seed,use_prefix=prefix,**kwargs)

        
        return self.generate.prompt2image(
            prompt,
            seed           = seed,
            sampler        = self.generate.sampler,
            steps          = opt.steps,
            cfg_scale      = opt.cfg_scale,
            ddim_eta       = self.generate.ddim_eta,
            width          = opt.width,
            height         = opt.height,
            init_img       = image,
            strength       = 0.83,
            image_callback = wrapped_callback,
            prefix         = prefix,
        )

    def _create_outpaint_image(self, image, direction_args):
        assert len(direction_args) in [1, 2], 'Direction (-D) must have exactly one or two arguments.'

        if len(direction_args) == 1:
            direction = direction_args[0]
            pixels = None
        elif len(direction_args) == 2:
            direction = direction_args[0]
            pixels = int(direction_args[1])

        assert direction in ['top', 'left', 'bottom', 'right'], 'Direction (-D) must be one of "top", "left", "bottom", "right"'

        image = image.convert("RGBA")
        # we always extend top, but rotate to extend along the requested side
        if direction == 'left':
            image = image.transpose(Image.Transpose.ROTATE_270)
        elif direction == 'bottom':
            image = image.transpose(Image.Transpose.ROTATE_180)
        elif direction == 'right':
            image = image.transpose(Image.Transpose.ROTATE_90)

        pixels = image.height//2 if pixels is None else int(pixels)
        assert 0 < pixels < image.height, 'Direction (-D) pixels length must be in the range 0 - image.size'

        # the top part of the image is taken from the source image mirrored
        # coordinates (0,0) are the upper left corner of an image
        top = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM).convert("RGBA")
        top = top.crop((0, top.height - pixels, top.width, top.height))

        # setting all alpha of the top part to 0
        alpha = top.getchannel("A")
        alpha.paste(0, (0, 0, top.width, top.height))
        top.putalpha(alpha)

        # taking the bottom from the original image
        bottom = image.crop((0, 0, image.width, image.height - pixels))

        new_img = image.copy()
        new_img.paste(top, (0, 0))
        new_img.paste(bottom, (0, pixels))

        # create a 10% dither in the middle
        dither = min(image.height//10, pixels)
        for x in range(0, image.width, 2):
            for y in range(pixels - dither, pixels + dither):
                (r, g, b, a) = new_img.getpixel((x, y))
                new_img.putpixel((x, y), (r, g, b, 0))

        # let's rotate back again
        if direction == 'left':
            new_img = new_img.transpose(Image.Transpose.ROTATE_90)
        elif direction == 'bottom':
            new_img = new_img.transpose(Image.Transpose.ROTATE_180)
        elif direction == 'right':
            new_img = new_img.transpose(Image.Transpose.ROTATE_270)

        return new_img


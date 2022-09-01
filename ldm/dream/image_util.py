from math import sqrt, floor, ceil
from PIL import Image

class InitImageResizer():
    """Simple class to create resized copies of an Image while preserving the aspect ratio."""
    def __init__(self,Image):
        self.image = Image

    def resize(self,width=None,height=None) -> Image:
        """
        Return a copy of the image resized to fit within
        a box width x height. The aspect ratio is 
        maintained. If neither width nor height are provided, 
        then returns a copy of the original image. If one or the other is
        provided, then the other will be calculated from the
        aspect ratio.

        Everything is floored to the nearest multiple of 64 so
        that it can be passed to img2img()
        """
        im    = self.image
        
        ar = im.width/float(im.height)

        # Infer missing values from aspect ratio
        if not(width or height): # both missing
            width  = im.width
            height = im.height
        elif not height:           # height missing
            height = int(width/ar)
        elif not width:            # width missing
            width  = int(height*ar)

        # rw and rh are the resizing width and height for the image
        # they maintain the aspect ratio, but may not completelyl fill up
        # the requested destination size
        (rw,rh) = (width,int(width/ar)) if im.width>=im.height else (int(height*ar),height)

        #round everything to multiples of 64
        width,height,rw,rh = map(
            lambda x: x-x%64, (width,height,rw,rh)
        )

        # no resize necessary, but return a copy
        if im.width == width and im.height == height:
            return im.copy()
        
        # otherwise resize the original image so that it fits inside the bounding box
        resized_image = self.image.resize((rw,rh),resample=Image.Resampling.LANCZOS)
        return resized_image

def make_grid(image_list, rows=None, cols=None):
    image_cnt = len(image_list)
    if None in (rows, cols):
        rows = floor(sqrt(image_cnt))  # try to make it square
        cols = ceil(image_cnt / rows)
    width = image_list[0].width
    height = image_list[0].height

    grid_img = Image.new('RGB', (width * cols, height * rows))
    i = 0
    for r in range(0, rows):
        for c in range(0, cols):
            if i >= len(image_list):
                break
            grid_img.paste(image_list[i], (c * width, r * height))
            i = i + 1

    return grid_img


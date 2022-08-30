from PIL import Image

class InitImageResizer():
    """Simple class to create resized copies of an Image while preserving the aspect ratio."""
    def __init__(self,Image):
        self.image = Image

    def resize(self,width=None,height=None) -> Image:
        """
        Return a copy of the image resized to width x height.
        The aspect ratio is maintained, with any excess space
        filled using black borders (i.e. letterboxed). If
        neither width nor height are provided, then returns
        a copy of the original image. If one or the other is
        provided, then the other will be calculated from the
        aspect ratio.

        Everything is floored to the nearest multiple of 64 so
        that it can be passed to img2img()
        """
        im    = self.image

        if not(width or height):
            return im.copy()

        ar = im.width/im.height

        # Infer missing values from aspect ratio
        if not height:          # height missing
            height = int(width/ar)
        if not width:          # width missing
            width  = int(height*ar)

        # rw and rh are the resizing width and height for the image
        # they maintain the aspect ratio, but may not completelyl fill up
        # the requested destination size
        (rw,rh) = (width,int(width/ar)) if im.width>=im.height else (int(height*ar),width)

        #round everything to multiples of 64
        width,height,rw,rh = map(
            lambda x: x-x%64, (width,height,rw,rh)
            )

        # resize the original image so that it fits inside the dest
        resized_image = self.image.resize((rw,rh),resample=Image.Resampling.LANCZOS)

        # create new destination image of specified dimensions
        # and paste the resized image into it centered appropriately
        new_image = Image.new('RGB',(width,height))
        new_image.paste(resized_image,((width-rw)//2,(height-rh)//2))

        return new_image

            

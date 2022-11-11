#resizes and adds a black bar to all images in directory original

from PIL import Image, ImageOps

import os

directory = 'original'

for filename in os.listdir(directory):
    var1 = directory + '/' + filename
    os.mkdir('E:/convert/original/' + filename)
    for i in os.listdir(var1):
        var4 = var1 + '/'
        var2 = var1 + '/' + i
        if os.path.isfile(var2):
           print(var2)
           im = Image.open(var2)
           im = ImageOps.pad(im, (512, 512), color='black')
           im.save('E:/convert/' + var2)

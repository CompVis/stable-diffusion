import webdataset as wds
from tqdm import tqdm
import numpy as np

import skimage
import numpy as np
from skimage.morphology import reconstruction

def generate_cell_mask(img):
    # Create the nuclei mask
    blurred_image = skimage.filters.gaussian(img[:, :, 2], sigma=2.0)
    t = skimage.filters.threshold_otsu(blurred_image)
    nuclei_binary_mask = blurred_image > t

    # Fill holes in the nuclei
    seed = np.copy(nuclei_binary_mask)
    seed[1:-1, 1:-1] = nuclei_binary_mask.max()
    nuclei_filled = reconstruction(seed, nuclei_binary_mask, method='erosion')
    nuclei_mask = nuclei_filled > 0

    # Create the cell body image
    gray_image = img[:, :, 0] + img[:, :, 1] + nuclei_filled + img[:, :, 3] > 0
    # blur the image to denoise
    blurred_image = skimage.filters.gaussian(gray_image, sigma=2.0)
    t = skimage.filters.threshold_otsu(blurred_image)
    binary_mask = blurred_image > t  

    return np.stack([binary_mask, nuclei_mask], axis=2) # dtype = bool

url = "/data/wei/hpa-webdataset-all-composite/webdataset_img.tar"
dataset = wds.WebDataset(url).decode().to_tuple("__key__", "img.pyd")
with open("logs/error-log-mask.txt", "w") as log:
    with wds.TarWriter('/data/wei/hpa-webdataset-all-composite/webdataset_mask.tar') as sink:
        for idx, data in tqdm(enumerate(dataset)):
            img = data[1]
            sink.write({
                "__key__": data[0],
                "mask.pyd": generate_cell_mask(img)
            })

# dataset = wds.WebDataset('/data/wei/hpa-webdataset-all-composite/webdataset_mask.tar').decode().to_tuple("__key__", "mask.pyd")
# print(next(iter(dataset))[1].shape)
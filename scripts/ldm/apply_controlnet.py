import copy
from PIL import Image, ImageOps
import numpy as np
import torch


# returns a conditioning with a controlnet applied to it, ready to pass it to a KSampler
def apply_controlnet(conditioning, control_net, image, strength):
    if strength == 0:
        return (conditioning,)

    c = []
    control_hint = image.movedim(-1, 1)
    for t in conditioning:
        n = [t[0], t[1].copy()]
        c_net = control_net.copy().set_cond_hint(control_hint, strength)
        if "control" in t[1]:
            c_net.set_previous_controlnet(t[1]["control"])
        n[1]["control"] = c_net
        n[1]["control_apply_to_uncond"] = True
        c.append(n)
    return (c,)


def load_image(image_path):
    i = Image.open(image_path)
    i = ImageOps.exif_transpose(i)
    image = i.convert("RGB")
    image = np.array(image).astype(np.float32) / 255.0
    image = torch.from_numpy(image)[None,]
    if "A" in i.getbands():
        mask = np.array(i.getchannel("A")).astype(np.float32) / 255.0
        mask = 1.0 - torch.from_numpy(mask)
    else:
        mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
    return (image, mask.unsqueeze(0))

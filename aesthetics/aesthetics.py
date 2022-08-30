import webdataset as wds
from PIL import Image
import io
import matplotlib.pyplot as plt
import os
import json

from warnings import filterwarnings


os.environ["CUDA_VISIBLE_DEVICES"] = "1"    # choose GPU if you are on a multi GPU server
import numpy as np
import torch
import pytorch_lightning as pl
import torch.nn as nn
from torchvision import datasets, transforms
import tqdm

from os.path import join
from datasets import load_dataset
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import json

import clip


from PIL import Image, ImageFile


#####  This script will predict the aesthetic score for this image file:

img_path = "../250k_data-0/img/000baa665498e7a61130d7662f81e698.jpg"





# if you changed the MLP architecture during training, change it also here:
class MLP(pl.LightningModule):
    def __init__(self, input_size, xcol='emb', ycol='avg_rating'):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            #nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, 16),
            #nn.ReLU(),

            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
            x = batch[self.xcol]
            y = batch[self.ycol].reshape(-1, 1)
            x_hat = self.layers(x)
            loss = F.mse_loss(x_hat, y)
            return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

def normalized(a, axis=-1, order=2):
    import numpy as np  # pylint: disable=import-outside-toplevel

    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


model = MLP(768)  # CLIP embedding dim is 768 for CLIP ViT L 14

s = torch.load("sac+logos+ava1-l14-linearMSE.pth")   # load the model you trained previously or the model available in this repo

model.load_state_dict(s)

model.to("cuda")
model.eval()


device = "cuda" if torch.cuda.is_available() else "cpu"
model2, preprocess = clip.load("ViT-L/14", device=device)  #RN50x64   

@torch.inference_mode()
def aesthetic(img_path):
    pil_image = Image.open(img_path)
    image = preprocess(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model2.encode_image(image)
    im_emb_arr = normalized(image_features.cpu().detach().numpy())
    prediction = model(torch.from_numpy(im_emb_arr).to(device).type(torch.cuda.FloatTensor))
    return prediction.item()

import json
import glob
import shutil

imdir = '../250k_data-0/img/'
ext = ['png', 'jpg', 'jpeg', 'bmp']
images = []
[images.extend(glob.glob(imdir + '*.' + e)) for e in ext]

aesthetic_scores = {}

try:
    for i in tqdm.tqdm(images):
        try:
            score = aesthetic(i)
        except:
            print(f'skipping {i}')
            continue
        if score < 5.0:
            shutil.move(i, i.replace('img', 'nonaesthetic'))
        elif score > 6.0:
            shutil.move(i, i.replace('img', 'aesthetic'))
        aesthetic_scores[i] = score
except KeyboardInterrupt:
    pass
finally:
    with open('scores.json', 'w') as f:
        f.write(json.dumps(aesthetic_scores))

#!/usr/bin/env python3
# Copyright (c) 2022 Lincoln D. Stein (https://github.com/lstein)
# Before running stable-diffusion on an internet-isolated machine,
# run this script from one with internet connectivity. The
# two machines must share a common .cache directory.
from transformers import CLIPTokenizer, CLIPTextModel
import clip
from transformers import BertTokenizerFast
import sys
import transformers
import os
import warnings
import urllib.request

transformers.logging.set_verbosity_error()

# this will preload the Bert tokenizer fles
print('preloading bert tokenizer...')

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
print('...success')

# this will download requirements for Kornia
print('preloading Kornia requirements (ignore the deprecation warnings)...')
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    import kornia
print('...success')

version = 'openai/clip-vit-large-patch14'

print('preloading CLIP model (Ignore the deprecation warnings)...')
sys.stdout.flush()

tokenizer = CLIPTokenizer.from_pretrained(version)
transformer = CLIPTextModel.from_pretrained(version)
print('\n\n...success')

# In the event that the user has installed GFPGAN and also elected to use
# RealESRGAN, this will attempt to download the model needed by RealESRGANer
gfpgan = False
try:
    from realesrgan import RealESRGANer

    gfpgan = True
except ModuleNotFoundError:
    pass

if gfpgan:
    print('Loading models from RealESRGAN and facexlib')
    try:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from facexlib.utils.face_restoration_helper import FaceRestoreHelper

        RealESRGANer(
            scale=2,
            model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
            model=RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=2,
            ),
        )

        RealESRGANer(
            scale=4,
            model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
            model=RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=4,
            ),
        )

        FaceRestoreHelper(1, det_model='retinaface_resnet50')
        print('...success')
    except Exception:
        import traceback
        print('Error loading ESRGAN:')
        print(traceback.format_exc())

    try:
        import urllib.request
        model_url  = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth'
        model_dest = 'src/gfpgan/experiments/pretrained_models/GFPGANv1.3.pth'

        if not os.path.exists(model_dest):
            print('downloading gfpgan model file...')
            urllib.request.urlretrieve(model_path,model_dest)
    except Exception:
        import traceback
        print('Error loading GFPGAN:')
        print(traceback.format_exc())
print('...success')

print('preloading CodeFormer model file...')
try:
        import urllib.request
        model_url  = 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth'
        model_dest = 'ldm/restoration/codeformer/weights/codeformer.pth'
        if not os.path.exists(model_dest):
            print('downloading codeformer model file...')
            os.makedirs(os.path.dirname(model_dest), exist_ok=True)
            urllib.request.urlretrieve(model_path,model_dest)
except Exception:
    import traceback
    print('Error loading CodeFormer:')
    print(traceback.format_exc())
print('...success')

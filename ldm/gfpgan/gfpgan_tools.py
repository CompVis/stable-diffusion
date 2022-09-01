import torch
import warnings
import os
import sys
import numpy as np

from PIL import Image
from scripts.dream import create_argv_parser

arg_parser = create_argv_parser()
opt = arg_parser.parse_args()

model_path = os.path.join(opt.gfpgan_dir, opt.gfpgan_model_path)
gfpgan_model_exists = os.path.isfile(model_path)

def _run_gfpgan(image, strength, prompt, seed, upsampler_scale=4):
    print(f'>> GFPGAN - Restoring Faces: {prompt} : seed:{seed}')
    gfpgan = None
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        warnings.filterwarnings('ignore', category=UserWarning)

        try:
            if not gfpgan_model_exists:
                raise Exception('GFPGAN model not found at path ' + model_path)

            sys.path.append(os.path.abspath(opt.gfpgan_dir))
            from gfpgan import GFPGANer

            bg_upsampler = _load_gfpgan_bg_upsampler(
                opt.gfpgan_bg_upsampler, upsampler_scale, opt.gfpgan_bg_tile
            )

            gfpgan = GFPGANer(
                model_path=model_path,
                upscale=upsampler_scale,
                arch='clean',
                channel_multiplier=2,
                bg_upsampler=bg_upsampler,
            )
        except Exception:
            import traceback

            print('>> Error loading GFPGAN:', file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)

    if gfpgan is None:
        print(
            f'>> GFPGAN not initialized, it must be loaded via the --gfpgan argument'
        )
        return image

    image = image.convert('RGB')

    cropped_faces, restored_faces, restored_img = gfpgan.enhance(
        np.array(image, dtype=np.uint8),
        has_aligned=False,
        only_center_face=False,
        paste_back=True,
    )
    res = Image.fromarray(restored_img)

    if strength < 1.0:
        # Resize the image to the new image if the sizes have changed
        if restored_img.size != image.size:
            image = image.resize(res.size)
        res = Image.blend(image, res, strength)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gfpgan = None

    return res


def _load_gfpgan_bg_upsampler(bg_upsampler, upsampler_scale, bg_tile=400):
    if bg_upsampler == 'realesrgan':
        if not torch.cuda.is_available():  # CPU
            warnings.warn(
                'The unoptimized RealESRGAN is slow on CPU. We do not use it. '
                'If you really want to use it, please modify the corresponding codes.'
            )
            bg_upsampler = None
        else:
            model_path = {
                2: 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
                4: 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
            }

            if upsampler_scale not in model_path:
                return None

            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer

            if upsampler_scale == 4:
                model = RRDBNet(
                    num_in_ch=3,
                    num_out_ch=3,
                    num_feat=64,
                    num_block=23,
                    num_grow_ch=32,
                    scale=4,
                )
            if upsampler_scale == 2:
                model = RRDBNet(
                    num_in_ch=3,
                    num_out_ch=3,
                    num_feat=64,
                    num_block=23,
                    num_grow_ch=32,
                    scale=2,
                )

            bg_upsampler = RealESRGANer(
                scale=upsampler_scale,
                model_path=model_path[upsampler_scale],
                model=model,
                tile=bg_tile,
                tile_pad=10,
                pre_pad=0,
                half=True,
            )  # need to set False in CPU mode
    else:
        bg_upsampler = None

    return bg_upsampler


def real_esrgan_upscale(image, strength, upsampler_scale, prompt, seed):
    print(
        f'>> Real-ESRGAN Upscaling: {prompt} : seed:{seed} : scale:{upsampler_scale}x'
    )

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        warnings.filterwarnings('ignore', category=UserWarning)

        try:
            upsampler = _load_gfpgan_bg_upsampler(
                opt.gfpgan_bg_upsampler, upsampler_scale, opt.gfpgan_bg_tile
            )
        except Exception:
            import traceback

            print('>> Error loading Real-ESRGAN:', file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)

    output, img_mode = upsampler.enhance(
        np.array(image, dtype=np.uint8),
        outscale=upsampler_scale,
        alpha_upsampler=opt.gfpgan_bg_upsampler,
    )

    res = Image.fromarray(output)

    if strength < 1.0:
        # Resize the image to the new image if the sizes have changed
        if output.size != image.size:
            image = image.resize(res.size)
        res = Image.blend(image, res, strength)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    upsampler = None

    return res

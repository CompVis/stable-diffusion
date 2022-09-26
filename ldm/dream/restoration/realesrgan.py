import torch
import warnings
import numpy as np

from PIL import Image


class ESRGAN():
    def __init__(self, bg_tile_size=400) -> None:
        self.bg_tile_size = bg_tile_size

        if not torch.cuda.is_available():  # CPU or MPS on M1
            use_half_precision = False
        else:
            use_half_precision = True

    def load_esrgan_bg_upsampler(self):
        if not torch.cuda.is_available():  # CPU or MPS on M1
            use_half_precision = False
        else:
            use_half_precision = True

        from realesrgan.archs.srvgg_arch import SRVGGNetCompact
        from realesrgan import RealESRGANer

        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
        model_path = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth'
        scale = 4

        bg_upsampler = RealESRGANer(
            scale=scale,
            model_path=model_path,
            model=model,
            tile=self.bg_tile_size,
            tile_pad=10,
            pre_pad=0,
            half=use_half_precision,
        )

        return bg_upsampler

    def process(self, image, strength: float, seed: str = None, upsampler_scale: int = 2):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=DeprecationWarning)
            warnings.filterwarnings('ignore', category=UserWarning)

            try:
                upsampler = self.load_esrgan_bg_upsampler()
            except Exception:
                import traceback
                import sys
                print('>> Error loading Real-ESRGAN:', file=sys.stderr)
                print(traceback.format_exc(), file=sys.stderr)

        if upsampler_scale == 0:
            print('>> Real-ESRGAN: Invalid scaling option. Image not upscaled.')
            return image

        if seed is not None:
            print(
                f'>> Real-ESRGAN Upscaling seed:{seed} : scale:{upsampler_scale}x'
            )
        
        output, _ = upsampler.enhance(
            np.array(image, dtype=np.uint8),
            outscale=upsampler_scale,
            alpha_upsampler='realesrgan',
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

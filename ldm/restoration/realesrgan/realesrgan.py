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

    def load_esrgan_bg_upsampler(self, upsampler_scale):
        if not torch.cuda.is_available():  # CPU or MPS on M1
            use_half_precision = False
        else:
            use_half_precision = True

        model_path = {
            2: 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
            4: 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
        }

        if upsampler_scale not in model_path:
            return None
        else:
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
                tile=self.bg_tile_size,
                tile_pad=10,
                pre_pad=0,
                half=use_half_precision,
            )

        return bg_upsampler

    def process(self, image, strength: float, seed: str = None, upsampler_scale: int = 2):
        if seed is not None:
            print(
                f'>> Real-ESRGAN Upscaling seed:{seed} : scale:{upsampler_scale}x'
            )

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=DeprecationWarning)
            warnings.filterwarnings('ignore', category=UserWarning)

            try:
                upsampler = self.load_esrgan_bg_upsampler(upsampler_scale)
            except Exception:
                import traceback
                import sys

                print('>> Error loading Real-ESRGAN:', file=sys.stderr)
                print(traceback.format_exc(), file=sys.stderr)

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

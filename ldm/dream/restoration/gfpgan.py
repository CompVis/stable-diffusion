import torch
import warnings
import os
import sys
import numpy as np

from PIL import Image


class GFPGAN():
    def __init__(
            self,
            gfpgan_dir='src/gfpgan',
            gfpgan_model_path='experiments/pretrained_models/GFPGANv1.4.pth') -> None:

        self.model_path = os.path.join(gfpgan_dir, gfpgan_model_path)
        self.gfpgan_model_exists = os.path.isfile(self.model_path)

        if not self.gfpgan_model_exists:
            print('## NOT FOUND: GFPGAN model not found at ' + self.model_path)
            return None
        sys.path.append(os.path.abspath(gfpgan_dir))

    def model_exists(self):
        return os.path.isfile(self.model_path)

    def process(self, image, strength: float, seed: str = None):
        if seed is not None:
            print(f'>> GFPGAN - Restoring Faces for image seed:{seed}')

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=DeprecationWarning)
            warnings.filterwarnings('ignore', category=UserWarning)
            try:
                from gfpgan import GFPGANer
                self.gfpgan = GFPGANer(
                    model_path=self.model_path,
                    upscale=1,
                    arch='clean',
                    channel_multiplier=2,
                    bg_upsampler=None,
                )
            except Exception:
                import traceback
                print('>> Error loading GFPGAN:', file=sys.stderr)
                print(traceback.format_exc(), file=sys.stderr)

        if self.gfpgan is None:
            print(
                f'>> WARNING: GFPGAN not initialized.'
            )
            print(
                f'>> Download https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth to {self.model_path}, \nor change GFPGAN directory with --gfpgan_dir.'
            )

        image = image.convert('RGB')

        _, _, restored_img = self.gfpgan.enhance(
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
        self.gfpgan = None

        return res

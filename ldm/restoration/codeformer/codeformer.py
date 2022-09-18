import os
import torch
import numpy as np
import warnings

pretrained_model_url = 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth'

class CodeFormerRestoration():
    def __init__(self) -> None:
        pass

    def process(self, image, strength, device, seed=None, fidelity=0.75):
        if seed is not None:
            print(f'>> CodeFormer - Restoring Faces for image seed:{seed}')
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=DeprecationWarning)
            warnings.filterwarnings('ignore', category=UserWarning)

            from basicsr.utils.download_util import load_file_from_url
            from basicsr.utils import img2tensor, tensor2img
            from facexlib.utils.face_restoration_helper import FaceRestoreHelper
            from ldm.restoration.codeformer.codeformer_arch import CodeFormer
            from torchvision.transforms.functional import normalize
            from PIL import Image
            
            cf_class = CodeFormer
            
            cf = cf_class(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9, connect_list=['32', '64', '128', '256']).to(device)
            
            checkpoint_path = load_file_from_url(url=pretrained_model_url, model_dir=os.path.abspath('ldm/restoration/codeformer/weights'), progress=True)
            checkpoint = torch.load(checkpoint_path)['params_ema']
            cf.load_state_dict(checkpoint)
            cf.eval()

            image = image.convert('RGB')

            face_helper = FaceRestoreHelper(upscale_factor=1, use_parse=True, device=device)
            face_helper.clean_all()
            face_helper.read_image(np.array(image, dtype=np.uint8))
            face_helper.get_face_landmarks_5(resize=640, eye_dist_threshold=5)
            face_helper.align_warp_face()

            for idx, cropped_face in enumerate(face_helper.cropped_faces):
                cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
                normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
                cropped_face_t = cropped_face_t.unsqueeze(0).to(device)

                try:
                    with torch.no_grad():
                        output = cf(cropped_face_t, w=fidelity, adain=True)[0]
                        restored_face = tensor2img(output.squeeze(0), rgb2bgr=True, min_max=(-1, 1))
                    del output
                    torch.cuda.empty_cache()
                except RuntimeError as error:
                    print(f'\tFailed inference for CodeFormer: {error}.')
                    restored_face = cropped_face

                restored_face = restored_face.astype('uint8')
                face_helper.add_restored_face(restored_face)


            face_helper.get_inverse_affine(None)

            restored_img = face_helper.paste_faces_to_input_image()

            res = Image.fromarray(restored_img)

            if strength < 1.0:
                # Resize the image to the new image if the sizes have changed
                if restored_img.size != image.size:
                    image = image.resize(res.size)
                res = Image.blend(image, res, strength)

            cf = None

            return res
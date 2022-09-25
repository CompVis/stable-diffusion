class Restoration():
    def __init__(self) -> None:
        pass

    def load_face_restore_models(self, gfpgan_dir='./src/gfpgan', gfpgan_model_path='experiments/pretrained_models/GFPGANv1.4.pth'):
        # Load GFPGAN
        gfpgan = self.load_gfpgan(gfpgan_dir, gfpgan_model_path)
        if gfpgan.gfpgan_model_exists:
            print('>> GFPGAN Initialized')
        else:
            print('>> GFPGAN Disabled')
            gfpgan = None
        
        # Load CodeFormer
        codeformer = self.load_codeformer()
        if codeformer.codeformer_model_exists:
            print('>> CodeFormer Initialized')
        else:
            print('>> CodeFormer Disabled')
            codeformer = None
        
        return gfpgan, codeformer

    # Face Restore Models
    def load_gfpgan(self, gfpgan_dir, gfpgan_model_path):
        from ldm.dream.restoration.gfpgan import GFPGAN
        return GFPGAN(gfpgan_dir, gfpgan_model_path)

    def load_codeformer(self):
        from ldm.dream.restoration.codeformer import CodeFormerRestoration
        return CodeFormerRestoration()

    # Upscale Models
    def load_esrgan(self, esrgan_bg_tile=400):
        from ldm.dream.restoration.realesrgan import ESRGAN
        esrgan = ESRGAN(esrgan_bg_tile)
        print('>> ESRGAN Initialized')
        return esrgan;

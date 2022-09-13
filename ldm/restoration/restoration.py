class Restoration():
    def __init__(self, gfpgan_dir='./src/gfpgan', gfpgan_model_path='experiments/pretrained_models/GFPGANv1.3.pth', esrgan_bg_tile=400) -> None:
        self.gfpgan_dir = gfpgan_dir
        self.gfpgan_model_path = gfpgan_model_path
        self.esrgan_bg_tile = esrgan_bg_tile

    def load_face_restore_models(self):
        # Load GFPGAN
        gfpgan = self.load_gfpgan()
        if gfpgan.gfpgan_model_exists:
            print('>> GFPGAN Initialized')
        
        # Load CodeFormer
        codeformer = self.load_codeformer()
        if codeformer.codeformer_model_exists:
            print('>> CodeFormer Initialized')
        
        return gfpgan, codeformer

    # Face Restore Models
    def load_gfpgan(self):
        from ldm.restoration.gfpgan.gfpgan import GFPGAN
        return GFPGAN(self.gfpgan_dir, self.gfpgan_model_path)

    def load_codeformer(self):
        from ldm.restoration.codeformer.codeformer import CodeFormerRestoration
        return CodeFormerRestoration()

    # Upscale Models
    def load_ersgan(self):
        from ldm.restoration.realesrgan.realesrgan import ESRGAN
        esrgan = ESRGAN(self.esrgan_bg_tile)
        print('>> ESRGAN Initialized')
        return esrgan;
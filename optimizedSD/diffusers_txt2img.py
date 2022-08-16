import torch
from diffusers import LDMTextToImagePipeline

pipe = LDMTextToImagePipeline.from_pretrained("CompVis/stable-diffusion-v1-3-diffusers", use_auth_token=True)

prompt  = "19th Century wooden engraving of Elon musk"

seed = torch.manual_seed(1024)
images = pipe([prompt], batch_size=1, num_inference_steps=50, guidance_scale=7, generator=seed,torch_device="cpu" )["sample"]

# save images
for idx, image in enumerate(images):
    image.save(f"image-{idx}.png")

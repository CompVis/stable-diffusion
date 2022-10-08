import clip
import glob
from PIL import Image
import torch
import tqdm

# Just put your images in a folder inside reference_images/
aesthetic_style = "aivazovsky"
image_paths = glob.glob(f"reference_images/{aesthetic_style}/*")


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14", device=device)


with torch.no_grad():
    embs = []
    for path in tqdm.tqdm(image_paths):
        image = preprocess(Image.open(path)).unsqueeze(0).to(device)
        emb = model.encode_image(image)
        embs.append(emb.cpu())

    embs = torch.cat(embs, dim=0).mean(dim=0, keepdim=True)

    # The generated embedding will be located here
    torch.save(embs, f"aesthetic_embeddings/{aesthetic_style}.pt")

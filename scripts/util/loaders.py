from PIL import Image

def load_pil_image(path: str) -> Image:
    """Load a PIL image from a file path."""
    return Image.open(path)
import os
import time
import warnings

from scripts.retro_diffusion import rd
from scripts.util.audio import play
from scripts.util.clbar import clbar
from PIL import Image

from rembg import remove

def rembg(numFiles):
    timer = time.time()
    files = []

    rd.logger(f"\n[#48a971]Removing [#48a971]{numFiles}[white] backgrounds")

    # Create a list of file paths
    for n in range(numFiles):
        files.append(f"temp/input{n+1}.png")

    # Process each file in the list
    for file in clbar(
        files,
        name="Processed",
        position="",
        unit="image",
        prefixwidth=12,
        suffixwidth=28,
    ):
        img = Image.open(file).convert("RGB")

        # Check if the file exists
        if os.path.isfile(file):
            # Ignore warnings during background removal
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                # Remove the background and save the image
                remove(img).save(file)

            if file != files[-1]:
                play("iteration.wav")
            else:
                play("batch.wav")
    rd.logger(
        f"[#c4f129]Removed [#48a971]{len(files)}[#c4f129] backgrounds in [#48a971]{round(time.time()-timer, 2)}[#c4f129] seconds"
    )
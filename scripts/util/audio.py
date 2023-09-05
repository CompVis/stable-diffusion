import os
import threading
import warnings
from winsound import PlaySound
from scripts.retro_diffusion import rd

def audioThread(file):
    try:
        absoluteFile = os.path.abspath(f"../sounds/{file}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            PlaySound.playsound(absoluteFile)
    except:
        pass


def play(file):
    if rd.sounds:
        try:
            threading.Thread(target=audioThread, args=(file,), daemon=True).start()
        except:
            pass
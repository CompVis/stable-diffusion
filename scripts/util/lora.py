import os
from typing import Dict, List

from scripts.util.crypto import decrypt, encrypt
from scripts.retro_diffusion import rd

from sdkit.models import load_model as sdkit_load_model
from sdkit.models import unload_model as sdkit_unload_model

def replace_extension(filename: str, new_extension: str) -> str:
    base_name = os.path.splitext(filename)[0]
    return f"{base_name}.{new_extension}"

def prepare_loras_for_inference(lora_path:str, loras: List[str], lora_weights: List[int]) -> List[Dict[str, float]]:
    loaded_loras = []
    
    for i, loraFile in enumerate(loras):
        if loraFile != "none":
            lora_filename = os.path.join(lora_path, loraFile)
            new_filename = replace_extension(lora_filename, "safetensors")
            
            if os.path.splitext(loraFile)[1] == ".pxlm":
                with open(lora_filename, "rb") as enc_file:
                    encrypted = enc_file.read()
                try:
                    decrypted = decrypt(encrypted)
                except:
                    decrypted = encrypted
                with open(lora_filename, "wb") as dec_file:
                    dec_file.write(decrypted)
                    
                # Replace extension
                os.rename(lora_filename, new_filename)
                    
            multiplier = lora_weights[i] / 100
            loaded_loras.append({ "path": new_filename, "multiplier": multiplier })
            
            # Set loras to rd context
            rd.loras = loaded_loras
            
            rd.logger(
                f"[#494b9b]Using [#48a971]{os.path.splitext(loraFile)[0]} [#494b9b]LoRA with [#48a971]{lora_weights[i]}% [#494b9b]strength"
            )
            
    
    return loaded_loras

def load_loras(loras: List[Dict[str, float]]): # loras: List of { path: str, multiplier: float }
    print("TODO: Support multiple loras")
    
    # check if contains elements
    if len(loras) == 0:
        return None
    
    lora = loras[0]
    path = lora["path"]
    multiplier = lora["multiplier"]
    
    rd.context.model_paths["lora"] = path
    rd.lora_alpha = multiplier
    
    print("Loading LoRA...")
    sdkit_load_model(rd.context, "lora")
    
def restore_loras_after_inference(loras: List[Dict[str, float]]):
    # Unload loras
    sdkit_unload_model(rd.context, "lora")
    
    # Clean loras in context
    rd.context.model_paths["lora"] = None
    
    print("Restoring LoRA...")
    print(f"loras: {loras}")
    
    # Encrypt loras and replace extension
    for lora in loras:
        lora_path = lora["path"]
        
        if os.path.splitext(lora_path)[1] == ".safetensors":
            with open(lora_path, "rb") as enc_file:
                decrypted = enc_file.read()
                
            encrypted = encrypt(decrypted)
            
            with open(lora_path, "wb") as dec_file:
                dec_file.write(encrypted)
                
            # Replace extension
            os.rename(lora_path, replace_extension(lora_path, "pxlm"))
    
    # Clean loras in rd context
    rd.loras = []
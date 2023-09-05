import os

from scripts.util.lora import replace_extension


def prepare_model_for_inference(model_path: str) -> str:
    # if model extension is .pxlm, replace it with .safetensors
    
    if os.path.splitext(model_path)[1] == ".pxlm":
        # Replace extension
        new_model_path = replace_extension(model_path, "safetensors")
        os.rename(model_path, new_model_path)
        return new_model_path
    else:
        # Return original model path
        return model_path
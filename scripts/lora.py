import os
import re
import io
from typing import Union
import torch
from safetensors.torch import load_file

re_unet_down_blocks = re.compile(r"lora_unet_down_blocks_(\d+)_attentions_(\d+)_(.+)")
re_unet_mid_blocks = re.compile(r"lora_unet_mid_block_attentions_(\d+)_(.+)")
re_unet_up_blocks = re.compile(r"lora_unet_up_blocks_(\d+)_attentions_(\d+)_(.+)")
re_text_block = re.compile(r"lora_te_text_model_encoder_layers_(\d+)_(.+)")

re_digits = re.compile(r"\d+")
re_x_proj = re.compile(r"(.*)_([qkv]_proj)$")
re_compiled = {}

suffix_conversion = {
    "attentions": {},
    "resnets": {
        "conv1": "in_layers_2",
        "conv2": "out_layers_3",
        "time_emb_proj": "emb_layers_1",
        "conv_shortcut": "skip_connection",
    }
}

checkpoint_dict_replacements = {
    'cond_stage_model.transformer.embeddings.': 'cond_stage_model.transformer.text_model.embeddings.',
    'cond_stage_model.transformer.encoder.': 'cond_stage_model.transformer.text_model.encoder.',
    'cond_stage_model.transformer.final_layer_norm.': 'cond_stage_model.transformer.text_model.final_layer_norm.',
}

loaded_loras = []

def convert_diffusers_name_to_compvis(key, is_sd2):
    def match(match_list, regex_text):
        regex = re_compiled.get(regex_text)
        if regex is None:
            regex = re.compile(regex_text)
            re_compiled[regex_text] = regex

        r = re.match(regex, key)
        if not r:
            return False

        match_list.clear()
        match_list.extend([int(x) if re.match(re_digits, x) else x for x in r.groups()])
        return True

    m = []

    if match(m, r"lora_unet_down_blocks_(\d+)_(attentions|resnets)_(\d+)_(.+)"):
        suffix = suffix_conversion.get(m[1], {}).get(m[3], m[3])
        return f"diffusion_model_input_blocks_{1 + m[0] * 3 + m[2]}_{1 if m[1] == 'attentions' else 0}_{suffix}"

    if match(m, r"lora_unet_mid_block_(attentions|resnets)_(\d+)_(.+)"):
        suffix = suffix_conversion.get(m[0], {}).get(m[2], m[2])
        return f"diffusion_model_middle_block_{1 if m[0] == 'attentions' else m[1] * 2}_{suffix}"

    if match(m, r"lora_unet_up_blocks_(\d+)_(attentions|resnets)_(\d+)_(.+)"):
        suffix = suffix_conversion.get(m[1], {}).get(m[3], m[3])
        return f"diffusion_model_output_blocks_{m[0] * 3 + m[2]}_{1 if m[1] == 'attentions' else 0}_{suffix}"

    if match(m, r"lora_unet_down_blocks_(\d+)_downsamplers_0_conv"):
        return f"diffusion_model_input_blocks_{3 + m[0] * 3}_0_op"

    if match(m, r"lora_unet_up_blocks_(\d+)_upsamplers_0_conv"):
        return f"diffusion_model_output_blocks_{2 + m[0] * 3}_{2 if m[0]>0 else 1}_conv"

    if match(m, r"lora_te_text_model_encoder_layers_(\d+)_(.+)"):
        if is_sd2:
            if 'mlp_fc1' in m[1]:
                return f"model_transformer_resblocks_{m[0]}_{m[1].replace('mlp_fc1', 'mlp_c_fc')}"
            elif 'mlp_fc2' in m[1]:
                return f"model_transformer_resblocks_{m[0]}_{m[1].replace('mlp_fc2', 'mlp_c_proj')}"
            else:
                return f"model_transformer_resblocks_{m[0]}_{m[1].replace('self_attn', 'attn')}"

        return f"transformer_text_model_encoder_layers_{m[0]}_{m[1]}"

    return key


class LoraOnDisk:
    def __init__(self, name, filename):
        self.name = name
        self.filename = filename


class LoraModule:
    def __init__(self, name):
        self.name = name
        self.multiplier = 1.0
        self.modules = {}
        self.mtime = None


class LoraUpDownModule:
    def __init__(self):
        self.up = None
        self.down = None

def get_state_dict_from_checkpoint(pl_sd):
    pl_sd = pl_sd.pop("state_dict", pl_sd)
    pl_sd.pop("state_dict", None)
    sd = {}
    for k, v in pl_sd.items():
        new_key = transform_checkpoint_dict_key(k)
        if new_key is not None:
            sd[new_key] = v
    pl_sd.clear()
    pl_sd.update(sd)
    return pl_sd

def transform_checkpoint_dict_key(k):
    for text, replacement in checkpoint_dict_replacements.items():
        if k.startswith(text):
            k = replacement + k[len(text):]
    return k

def assign_lora_names_to_compvis_modules(model, modelCS):
    if not hasattr(modelCS, 'cond_stage_model'):
        print("No cond_stage_model found in modelCS, skipping Lora")
        return
    
    lora_layer_mapping = {}

    for name, module in modelCS.cond_stage_model.named_modules():
        lora_name = name.replace(".", "_")
        lora_layer_mapping[lora_name] = module
        module.lora_layer_name = lora_name

    for name, module in model.model1.named_modules():
        lora_name = name.replace(".", "_")
        lora_layer_mapping[lora_name] = module
        module.lora_layer_name = lora_name
        
    for name, module in model.model2.named_modules():
        lora_name = name.replace(".", "_")
        lora_layer_mapping[lora_name] = module
        module.lora_layer_name = lora_name

    model.lora_layer_mapping = lora_layer_mapping
    print("Added LoRA layers")

def load_lora_raw(filename):
    return load_file(filename)

def load_lora(filename, model):
    lora_tensors = load_file(filename)
    lora = LoraModule(filename)
    lora.mtime = os.path.getmtime(filename)

    sd = get_state_dict_from_checkpoint(lora_tensors)

    keys_failed_to_match = []

    for key_diffusers, weight in sd.items():
        key_diffusers_without_lora_parts, lora_key = key_diffusers.split(".", 1)
        key = convert_diffusers_name_to_compvis(key_diffusers_without_lora_parts, False)

        sd_module = model.lora_layer_mapping.get(key, None)

        if sd_module is None:
            m = re_x_proj.match(key)
            if m:
                sd_module = model.lora_layer_mapping.get(m.group(1), None)

        if sd_module is None:
            keys_failed_to_match[key_diffusers] = key
            continue

        lora_module = lora.modules.get(key, None)
        if lora_module is None:
            lora_module = LoraUpDownModule()
            lora.modules[key] = lora_module

        if lora_key == "alpha":
            lora_module.alpha = weight.item()
            continue

        if type(sd_module) == torch.nn.Linear:
            module = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False)
        elif type(sd_module) == torch.nn.modules.linear.NonDynamicallyQuantizableLinear:
            module = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False)
        elif type(sd_module) == torch.nn.MultiheadAttention:
            module = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False)
        elif type(sd_module) == torch.nn.Conv2d and weight.shape[2:] == (1, 1):
            module = torch.nn.Conv2d(weight.shape[1], weight.shape[0], (1, 1), bias=False)
        elif type(sd_module) == torch.nn.Conv2d and weight.shape[2:] == (3, 3):
            module = torch.nn.Conv2d(weight.shape[1], weight.shape[0], (3, 3), bias=False)
        else:
            print(f'Lora layer {key_diffusers} matched a layer with unsupported type: {type(sd_module).__name__}')
            continue

        with torch.no_grad():
            module.weight.copy_(weight)

        module.to(device=torch.device("cpu"), dtype=torch.float16)

        if lora_key == "lora_up.weight":
            lora_module.up = module
        elif lora_key == "lora_down.weight":
            lora_module.down = module
        else:
            raise AssertionError(f"Bad Lora layer name: {key_diffusers} - must end in lora_up.weight, lora_down.weight or alpha")

    if len(keys_failed_to_match) > 0:
        print(f"Failed to match keys when loading Lora {filename}: {keys_failed_to_match}")

    return lora

def lora_apply_weights(self: Union[torch.nn.Conv2d, torch.nn.Linear, torch.nn.MultiheadAttention]):
    """
    Applies the currently selected set of Loras to the weights of torch layer self.
    If weights already have this particular set of loras applied, does nothing.
    If not, restores orginal weights from backup and alters weights according to loras.
    """

    lora_layer_name = getattr(self, 'lora_layer_name', None)
    if lora_layer_name is None:
        return

    current_names = getattr(self, "lora_current_names", ())
    wanted_names = tuple((x.name, x.multiplier) for x in loaded_loras)

    weights_backup = getattr(self, "lora_weights_backup", None)
    if weights_backup is None:
        if isinstance(self, torch.nn.MultiheadAttention):
            weights_backup = (self.in_proj_weight.to(torch.device("cpu"), copy=True), self.out_proj.weight.to(torch.device("cpu"), copy=True))
        else:
            weights_backup = self.weight.to(torch.device("cpu"), copy=True)

        self.lora_weights_backup = weights_backup

    if current_names != wanted_names:
        lora_restore_weights_from_backup(self)

        for lora in loaded_loras:
            module = lora.modules.get(lora_layer_name, None)
            if module is not None and hasattr(self, 'weight'):
                self.weight += lora_calc_updown(lora, module, self.weight)
                continue

            module_q = lora.modules.get(lora_layer_name + "_q_proj", None)
            module_k = lora.modules.get(lora_layer_name + "_k_proj", None)
            module_v = lora.modules.get(lora_layer_name + "_v_proj", None)
            module_out = lora.modules.get(lora_layer_name + "_out_proj", None)

            if isinstance(self, torch.nn.MultiheadAttention) and module_q and module_k and module_v and module_out:
                updown_q = lora_calc_updown(lora, module_q, self.in_proj_weight)
                updown_k = lora_calc_updown(lora, module_k, self.in_proj_weight)
                updown_v = lora_calc_updown(lora, module_v, self.in_proj_weight)
                updown_qkv = torch.vstack([updown_q, updown_k, updown_v])

                self.in_proj_weight += updown_qkv
                self.out_proj.weight += lora_calc_updown(lora, module_out, self.out_proj.weight)
                continue

            if module is None:
                continue

            print(f'failed to calculate lora weights for layer {lora_layer_name}')

        self.lora_current_names = wanted_names

def lora_calc_updown(lora, module, target):
    with torch.no_grad():
        up = module.up.weight.to(target.device, dtype=target.dtype)
        down = module.down.weight.to(target.device, dtype=target.dtype)

        if up.shape[2:] == (1, 1) and down.shape[2:] == (1, 1):
            updown = (up.squeeze(2).squeeze(2) @ down.squeeze(2).squeeze(2)).unsqueeze(2).unsqueeze(3)
        elif up.shape[2:] == (3, 3) or down.shape[2:] == (3, 3):
            updown = torch.nn.functional.conv2d(down.permute(1, 0, 2, 3), up).permute(1, 0, 2, 3)
        else:
            updown = up @ down

        updown = updown * lora.multiplier * (module.alpha / module.up.weight.shape[1] if module.alpha else 1.0)

        return updown


def lora_restore_weights_from_backup(self: Union[torch.nn.Conv2d, torch.nn.Linear, torch.nn.MultiheadAttention]):
    weights_backup = getattr(self, "lora_weights_backup", None)

    if weights_backup is None:
        return

    if isinstance(self, torch.nn.MultiheadAttention):
        self.in_proj_weight.copy_(weights_backup[0])
        self.out_proj.weight.copy_(weights_backup[1])
    else:
        self.weight.copy_(weights_backup)


def lora_reset_cached_weight(self: Union[torch.nn.Conv2d, torch.nn.Linear]):
    self.lora_current_names = ()
    self.lora_weights_backup = None


def lora_Linear_forward(self, input):
    lora_apply_weights(self)

    return torch.nn.Linear_forward_before_lora(self, input)


def lora_Linear_load_state_dict(self, *args, **kwargs):
    lora_reset_cached_weight(self)

    return torch.nn.Linear_load_state_dict_before_lora(self, *args, **kwargs)


def lora_Conv2d_forward(self, input):
    lora_apply_weights(self)

    return torch.nn.Conv2d_forward_before_lora(self, input)


def lora_Conv2d_load_state_dict(self, *args, **kwargs):
    lora_reset_cached_weight(self)

    return torch.nn.Conv2d_load_state_dict_before_lora(self, *args, **kwargs)


def lora_MultiheadAttention_forward(self, *args, **kwargs):
    lora_apply_weights(self)

    return torch.nn.MultiheadAttention_forward_before_lora(self, *args, **kwargs)


def lora_MultiheadAttention_load_state_dict(self, *args, **kwargs):
    lora_reset_cached_weight(self)

    return torch.nn.MultiheadAttention_load_state_dict_before_lora(self, *args, **kwargs)


def unload():
    pass

def register_lora_for_inference(lora):
    loaded_loras.append(lora)
    
def remove_lora_for_inference(lora):
    loaded_loras.remove(lora)

def apply_lora():    
    if not hasattr(torch.nn, 'Linear_forward_before_lora'):
        torch.nn.Linear_forward_before_lora = torch.nn.Linear.forward

    if not hasattr(torch.nn, 'Linear_load_state_dict_before_lora'):
        torch.nn.Linear_load_state_dict_before_lora = torch.nn.Linear._load_from_state_dict

    if not hasattr(torch.nn, 'Conv2d_forward_before_lora'):
        torch.nn.Conv2d_forward_before_lora = torch.nn.Conv2d.forward

    if not hasattr(torch.nn, 'Conv2d_load_state_dict_before_lora'):
        torch.nn.Conv2d_load_state_dict_before_lora = torch.nn.Conv2d._load_from_state_dict

    if not hasattr(torch.nn, 'MultiheadAttention_forward_before_lora'):
        torch.nn.MultiheadAttention_forward_before_lora = torch.nn.MultiheadAttention.forward

    if not hasattr(torch.nn, 'MultiheadAttention_load_state_dict_before_lora'):
        torch.nn.MultiheadAttention_load_state_dict_before_lora = torch.nn.MultiheadAttention._load_from_state_dict

    torch.nn.Linear.forward = lora_Linear_forward
    torch.nn.Linear._load_from_state_dict = lora_Linear_load_state_dict
    torch.nn.Conv2d.forward = lora_Conv2d_forward
    torch.nn.Conv2d._load_from_state_dict = lora_Conv2d_load_state_dict
    torch.nn.MultiheadAttention.forward = lora_MultiheadAttention_forward
    torch.nn.MultiheadAttention._load_from_state_dict = lora_MultiheadAttention_load_state_dict
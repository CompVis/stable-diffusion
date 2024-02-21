import torch

import ldm.utils
import ldm.model_management
import ldm.model_detection
import ldm.model_patcher


class CLIP:
    def __init__(self, target=None, embedding_directory=None, no_init=False):
        if no_init:
            return
        params = target.params.copy()
        clip = target.clip
        tokenizer = target.tokenizer

        load_device = ldm.model_management.text_encoder_device()
        offload_device = ldm.model_management.text_encoder_offload_device()
        params["device"] = offload_device
        params["dtype"] = ldm.model_management.text_encoder_dtype(load_device)

        self.cond_stage_model = clip(**(params))

        self.tokenizer = tokenizer(embedding_directory=embedding_directory)
        self.patcher = ldm.model_patcher.ModelPatcher(
            self.cond_stage_model,
            load_device=load_device,
            offload_device=offload_device,
        )
        self.layer_idx = None

    def clone(self):
        n = CLIP(no_init=True)
        n.patcher = self.patcher.clone()
        n.cond_stage_model = self.cond_stage_model
        n.tokenizer = self.tokenizer
        n.layer_idx = self.layer_idx
        return n

    def add_patches(self, patches, strength_patch=1.0, strength_model=1.0):
        return self.patcher.add_patches(patches, strength_patch, strength_model)

    def clip_layer(self, layer_idx):
        self.layer_idx = layer_idx

    def tokenize(self, text, return_word_ids=False):
        return self.tokenizer.tokenize_with_weights(text, return_word_ids)

    def encode_from_tokens(self, tokens, return_pooled=False):
        if self.layer_idx is not None:
            self.cond_stage_model.clip_layer(self.layer_idx)
        else:
            self.cond_stage_model.reset_clip_layer()

        self.load_model()
        cond, pooled = self.cond_stage_model.encode_token_weights(tokens)
        if return_pooled:
            return cond, pooled
        return cond

    def encode(self, text):
        tokens = self.tokenize(text)
        return self.encode_from_tokens(tokens)

    def load_sd(self, sd):
        return self.cond_stage_model.load_sd(sd)

    def get_sd(self):
        return self.cond_stage_model.state_dict()

    def load_model(self):
        ldm.model_management.load_model_gpu(self.patcher)
        return self.patcher

    def get_key_patches(self):
        return self.patcher.get_key_patches()


def load_model_weights(model, sd):
    m, u = model.load_state_dict(sd, strict=False)
    m = set(m)
    unexpected_keys = set(u)

    k = list(sd.keys())
    for x in k:
        if x not in unexpected_keys:
            w = sd.pop(x)
            del w
    if len(m) > 0:
        print("missing", m)
    return model


def load_checkpoint_guess_config(
    ckpt_path,
    output_vae=True,
    output_clip=True,
    output_clipvision=False,
    embedding_directory=None,
    output_model=True,
):
    sd = ldm.utils.load_torch_file(ckpt_path)
    sd_keys = sd.keys()
    clip = None
    clipvision = None
    vae = None
    model = None
    model_patcher = None
    clip_target = None

    parameters = ldm.utils.calculate_parameters(sd, "model.diffusion_model.")
    unet_dtype = ldm.model_management.unet_dtype(model_params=parameters)
    load_device = ldm.model_management.get_torch_device()
    manual_cast_dtype = ldm.model_management.unet_manual_cast(unet_dtype, load_device)

    class WeightsLoader(torch.nn.Module):
        pass

    model_config = ldm.model_detection.model_config_from_unet(
        sd, "model.diffusion_model.", unet_dtype
    )
    model_config.set_manual_cast(manual_cast_dtype)

    if model_config is None:
        raise RuntimeError(
            "ERROR: Could not detect model type of: {}".format(ckpt_path)
        )

    # if model_config.clip_vision_prefix is not None:
    #     if output_clipvision:
    #         clipvision = clip_vision.load_clipvision_from_sd(
    #             sd, model_config.clip_vision_prefix, True
    #         )

    if output_model:
        inital_load_device = ldm.model_management.unet_inital_load_device(
            parameters, unet_dtype
        )
        offload_device = ldm.model_management.unet_offload_device()
        model = model_config.get_model(
            sd, "model.diffusion_model.", device=inital_load_device
        )
        model.load_model_weights(sd, "model.diffusion_model.")

    # if output_vae:
    #     vae_sd = ldm.utils.state_dict_prefix_replace(
    #         sd, {k: "" for k in model_config.vae_key_prefix}, filter_keys=True
    #     )
    #     vae_sd = model_config.process_vae_state_dict(vae_sd)
    #     vae = VAE(sd=vae_sd)

    if output_clip:
        w = WeightsLoader()
        clip_target = model_config.clip_target()
        if clip_target is not None:
            sd = model_config.process_clip_state_dict(sd)
            if any(k.startswith("cond_stage_model.") for k in sd):
                clip = CLIP(clip_target, embedding_directory=embedding_directory)
                w.cond_stage_model = clip.cond_stage_model
                load_model_weights(w, sd)
            else:
                print(
                    "no CLIP/text encoder weights in checkpoint, the text encoder model will not be loaded."
                )

    left_over = sd.keys()
    if len(left_over) > 0:
        # print("left over keys:", left_over)
        pass

    if output_model:
        model_patcher = ldm.model_patcher.ModelPatcher(
            model,
            load_device=load_device,
            offload_device=ldm.model_management.unet_offload_device(),
            current_device=inital_load_device,
        )
        if inital_load_device != torch.device("cpu"):
            ldm.model_management.load_model_gpu(model_patcher)

    return (model_patcher, clip, vae, clipvision)

import torch
import ldm.utils as utils
import ldm.supported_models_base as supported_models_base
import ldm.sd1_clip as sd1_clip
from ldm.latent_formats import LatentFormat_SD15


class SD15(supported_models_base.BASE):
    unet_config = {
        "context_dim": 768,
        "model_channels": 320,
        "use_linear_in_transformer": False,
        "adm_in_channels": None,
        "use_temporal_attention": False,
    }

    unet_extra_config = {
        "num_heads": 8,
        "num_head_channels": -1,
    }

    latent_format = LatentFormat_SD15

    def process_clip_state_dict(self, state_dict):
        k = list(state_dict.keys())
        for x in k:
            if x.startswith("cond_stage_model.transformer.") and not x.startswith(
                "cond_stage_model.transformer.text_model."
            ):
                y = x.replace(
                    "cond_stage_model.transformer.",
                    "cond_stage_model.transformer.text_model.",
                )
                state_dict[y] = state_dict.pop(x)

        if (
            "cond_stage_model.transformer.text_model.embeddings.position_ids"
            in state_dict
        ):
            ids = state_dict[
                "cond_stage_model.transformer.text_model.embeddings.position_ids"
            ]
            if ids.dtype == torch.float32:
                state_dict[
                    "cond_stage_model.transformer.text_model.embeddings.position_ids"
                ] = ids.round()

        replace_prefix = {}
        replace_prefix["cond_stage_model."] = "cond_stage_model.clip_l."
        state_dict = utils.state_dict_prefix_replace(state_dict, replace_prefix)
        return state_dict

    def process_clip_state_dict_for_saving(self, state_dict):
        replace_prefix = {"clip_l.": "cond_stage_model."}
        return utils.state_dict_prefix_replace(state_dict, replace_prefix)

    def clip_target(self):
        return supported_models_base.ClipTarget(
            sd1_clip.SD1Tokenizer, sd1_clip.SD1ClipModel
        )


models = [SD15]

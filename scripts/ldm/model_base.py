import torch
import torch
import ldm.conds
from enum import Enum

import ldm.ops
import ldm.model_management

from ldm.cldm_models import UNetModel
from . import utils

class ModelType(Enum):
    EPS = 1
    V_PREDICTION = 2
    V_PREDICTION_EDM = 3


from ldm.model_sampling import EPS, V_PREDICTION, ModelSamplingDiscrete, ModelSamplingContinuousEDM


def model_sampling(model_config, model_type):
    s = ModelSamplingDiscrete

    if model_type == ModelType.EPS:
        c = EPS
    elif model_type == ModelType.V_PREDICTION:
        c = V_PREDICTION
    elif model_type == ModelType.V_PREDICTION_EDM:
        c = V_PREDICTION
        s = ModelSamplingContinuousEDM

    class ModelSampling(s, c):
        pass

    return ModelSampling(model_config)


class BaseModel(torch.nn.Module):
    def __init__(
        self, model_config, model_type=ModelType.EPS, device=None, unet_model=UNetModel
    ):
        super().__init__()

        unet_config = model_config.unet_config
        self.latent_format = model_config.latent_format
        self.model_config = model_config
        self.manual_cast_dtype = model_config.manual_cast_dtype

        if not unet_config.get("disable_unet_model_creation", False):
            if self.manual_cast_dtype is not None:
                operations = ldm.ops.manual_cast
            else:
                operations = ldm.ops.disable_weight_init
            self.diffusion_model = unet_model(
                **unet_config, device=device, operations=operations
            )
        self.model_type = model_type
        self.model_sampling = model_sampling(model_config, model_type)

        self.adm_channels = unet_config.get("adm_in_channels", None)
        if self.adm_channels is None:
            self.adm_channels = 0
        self.inpaint_model = False
        print("model_type", model_type.name)
        print("adm", self.adm_channels)

    def apply_model(
        self,
        x,
        t,
        c_concat=None,
        c_crossattn=None,
        control=None,
        transformer_options={},
        **kwargs
    ):
        sigma = t
        xc = self.model_sampling.calculate_input(sigma, x)
        if c_concat is not None:
            xc = torch.cat([xc] + [c_concat], dim=1)

        context = c_crossattn
        dtype = self.get_dtype()

        if self.manual_cast_dtype is not None:
            dtype = self.manual_cast_dtype

        xc = xc.to(dtype)
        t = self.model_sampling.timestep(t).float()
        context = context.to(dtype)
        extra_conds = {}
        for o in kwargs:
            extra = kwargs[o]
            if hasattr(extra, "dtype"):
                if extra.dtype != torch.int and extra.dtype != torch.long:
                    extra = extra.to(dtype)
            extra_conds[o] = extra

        model_output = self.diffusion_model(
            xc,
            t,
            context=context,
            control=control,
            transformer_options=transformer_options,
            **extra_conds
        ).float()
        return self.model_sampling.calculate_denoised(sigma, model_output, x)

    def get_dtype(self):
        return self.diffusion_model.dtype

    def is_adm(self):
        return self.adm_channels > 0

    def encode_adm(self, **kwargs):
        return None

    def extra_conds(self, **kwargs):
        out = {}
        if self.inpaint_model:
            concat_keys = ("mask", "masked_image")
            cond_concat = []
            denoise_mask = kwargs.get("concat_mask", kwargs.get("denoise_mask", None))
            concat_latent_image = kwargs.get("concat_latent_image", None)
            if concat_latent_image is None:
                concat_latent_image = kwargs.get("latent_image", None)
            else:
                concat_latent_image = self.process_latent_in(concat_latent_image)

            noise = kwargs.get("noise", None)
            device = kwargs["device"]

            if concat_latent_image.shape[1:] != noise.shape[1:]:
                concat_latent_image = utils.common_upscale(
                    concat_latent_image,
                    noise.shape[-1],
                    noise.shape[-2],
                    "bilinear",
                    "center",
                )

            concat_latent_image = utils.resize_to_batch_size(
                concat_latent_image, noise.shape[0]
            )

            if len(denoise_mask.shape) == len(noise.shape):
                denoise_mask = denoise_mask[:, :1]

            denoise_mask = denoise_mask.reshape(
                (-1, 1, denoise_mask.shape[-2], denoise_mask.shape[-1])
            )
            if denoise_mask.shape[-2:] != noise.shape[-2:]:
                denoise_mask = utils.common_upscale(
                    denoise_mask, noise.shape[-1], noise.shape[-2], "bilinear", "center"
                )
            denoise_mask = utils.resize_to_batch_size(
                denoise_mask.round(), noise.shape[0]
            )

            def blank_inpaint_image_like(latent_image):
                blank_image = torch.ones_like(latent_image)
                # these are the values for "zero" in pixel space translated to latent space
                blank_image[:, 0] *= 0.8223
                blank_image[:, 1] *= -0.6876
                blank_image[:, 2] *= 0.6364
                blank_image[:, 3] *= 0.1380
                return blank_image

            for ck in concat_keys:
                if denoise_mask is not None:
                    if ck == "mask":
                        cond_concat.append(denoise_mask.to(device))
                    elif ck == "masked_image":
                        cond_concat.append(
                            concat_latent_image.to(device)
                        )  # NOTE: the latent_image should be masked by the mask in pixel space
                else:
                    if ck == "mask":
                        cond_concat.append(torch.ones_like(noise)[:, :1])
                    elif ck == "masked_image":
                        cond_concat.append(blank_inpaint_image_like(noise))
            data = torch.cat(cond_concat, dim=1)
            out["c_concat"] = ldm.conds.CONDNoiseShape(data)

        adm = self.encode_adm(**kwargs)
        if adm is not None:
            out["y"] = ldm.conds.CONDRegular(adm)

        cross_attn = kwargs.get("cross_attn", None)
        if cross_attn is not None:
            out["c_crossattn"] = ldm.conds.CONDCrossAttn(cross_attn)

        cross_attn_cnet = kwargs.get("cross_attn_controlnet", None)
        if cross_attn_cnet is not None:
            out["crossattn_controlnet"] = ldm.conds.CONDCrossAttn(cross_attn_cnet)

        return out

    def load_model_weights(self, sd, unet_prefix=""):
        to_load = {}
        keys = list(sd.keys())
        for k in keys:
            if k.startswith(unet_prefix):
                to_load[k[len(unet_prefix) :]] = sd.pop(k)

        to_load = self.model_config.process_unet_state_dict(to_load)
        m, u = self.diffusion_model.load_state_dict(to_load, strict=False)
        if len(m) > 0:
            print("unet missing:", m)

        if len(u) > 0:
            print("unet unexpected:", u)
        del to_load
        return self

    def process_latent_in(self, latent):
        return self.latent_format.process_in(latent)

    def process_latent_out(self, latent):
        return self.latent_format.process_out(latent)

    def state_dict_for_saving(
        self, clip_state_dict=None, vae_state_dict=None, clip_vision_state_dict=None
    ):
        extra_sds = []
        if clip_state_dict is not None:
            extra_sds.append(
                self.model_config.process_clip_state_dict_for_saving(clip_state_dict)
            )
        if vae_state_dict is not None:
            extra_sds.append(
                self.model_config.process_vae_state_dict_for_saving(vae_state_dict)
            )
        if clip_vision_state_dict is not None:
            extra_sds.append(
                self.model_config.process_clip_vision_state_dict_for_saving(
                    clip_vision_state_dict
                )
            )

        unet_state_dict = self.diffusion_model.state_dict()
        unet_state_dict = self.model_config.process_unet_state_dict_for_saving(
            unet_state_dict
        )

        if self.get_dtype() == torch.float16:
            extra_sds = map(
                lambda sd: utils.convert_sd_to(sd, torch.float16), extra_sds
            )

        if self.model_type == ModelType.V_PREDICTION:
            unet_state_dict["v_pred"] = torch.tensor([])

        for sd in extra_sds:
            unet_state_dict.update(sd)

        return unet_state_dict

    def set_inpaint(self):
        self.inpaint_model = True

    def memory_required(self, input_shape):
        if (
            ldm.model_management.xformers_enabled()
            or ldm.model_management.pytorch_attention_flash_attention()
        ):
            dtype = self.get_dtype()
            if self.manual_cast_dtype is not None:
                dtype = self.manual_cast_dtype
            # TODO: this needs to be tweaked
            area = input_shape[0] * input_shape[2] * input_shape[3]
            return (area * ldm.model_management.dtype_size(dtype) / 50) * (
                1024 * 1024
            )
        else:
            # TODO: this formula might be too aggressive since I tweaked the sub-quad and split algorithms to use less memory.
            area = input_shape[0] * input_shape[2] * input_shape[3]
            return (((area * 0.6) / 0.9) + 1024) * (1024 * 1024)

    def __init__(self, model_config, model_type=ModelType.EPS, device=None):
        super().__init__()

        unet_config = model_config.unet_config
        self.latent_format = model_config.latent_format
        self.model_config = model_config

        if not unet_config.get("disable_unet_model_creation", False):
            self.diffusion_model = UNetModel(**unet_config, device=device)
        self.model_type = model_type
        self.model_sampling = model_sampling(model_config, model_type)

        self.adm_channels = unet_config.get("adm_in_channels", None)
        if self.adm_channels is None:
            self.adm_channels = 0
        self.inpaint_model = False

    def apply_model(self, x, t, c_concat=None, c_crossattn=None, control=None, transformer_options={}, **kwargs):
        sigma = t
        xc = self.model_sampling.calculate_input(sigma, x)
        if c_concat is not None:
            xc = torch.cat([xc] + [c_concat], dim=1)

        context = c_crossattn
        dtype = self.get_dtype()
        xc = xc.to(dtype)
        t = self.model_sampling.timestep(t).float()
        context = context.to(dtype)
        extra_conds = {}
        for o in kwargs:
            extra = kwargs[o]
            if hasattr(extra, "to"):
                extra = extra.to(dtype)
            extra_conds[o] = extra
        model_output = self.diffusion_model(xc, t, context=context, control=control, transformer_options=transformer_options, **extra_conds).float()
        return self.model_sampling.calculate_denoised(sigma, model_output, x)

    def get_dtype(self):
        return self.diffusion_model.dtype

    def is_adm(self):
        return self.adm_channels > 0

    def encode_adm(self, **kwargs):
        return None

    def extra_conds(self, **kwargs):
        out = {}
        if self.inpaint_model:
            concat_keys = ("mask", "masked_image")
            cond_concat = []
            denoise_mask = kwargs.get("denoise_mask", None)
            latent_image = kwargs.get("latent_image", None)
            noise = kwargs.get("noise", None)
            device = kwargs["device"]

            def blank_inpaint_image_like(latent_image):
                blank_image = torch.ones_like(latent_image)
                # these are the values for "zero" in pixel space translated to latent space
                blank_image[:,0] *= 0.8223
                blank_image[:,1] *= -0.6876
                blank_image[:,2] *= 0.6364
                blank_image[:,3] *= 0.1380
                return blank_image

            for ck in concat_keys:
                if denoise_mask is not None:
                    if ck == "mask":
                        cond_concat.append(denoise_mask[:,:1].to(device))
                    elif ck == "masked_image":
                        cond_concat.append(latent_image.to(device)) #NOTE: the latent_image should be masked by the mask in pixel space
                else:
                    if ck == "mask":
                        cond_concat.append(torch.ones_like(noise)[:,:1])
                    elif ck == "masked_image":
                        cond_concat.append(blank_inpaint_image_like(noise))
            data = torch.cat(cond_concat, dim=1)
            out['c_concat'] = ldm.conds.CONDNoiseShape(data)
        adm = self.encode_adm(**kwargs)
        if adm is not None:
            out['y'] = ldm.conds.CONDRegular(adm)
        return out

    def load_model_weights(self, sd, unet_prefix=""):
        to_load = {}
        keys = list(sd.keys())
        for k in keys:
            if k.startswith(unet_prefix):
                to_load[k[len(unet_prefix):]] = sd.pop(k)

        to_load = self.model_config.process_unet_state_dict(to_load)
        m, u = self.diffusion_model.load_state_dict(to_load, strict=False)
        if len(m) > 0:
            print("unet missing:", m)

        if len(u) > 0:
            print("unet unexpected:", u)
        del to_load
        return self

    def process_latent_in(self, latent):
        return self.latent_format.process_in(latent)

    def process_latent_out(self, latent):
        return self.latent_format.process_out(latent)

    def state_dict_for_saving(self, clip_state_dict, vae_state_dict):
        clip_state_dict = self.model_config.process_clip_state_dict_for_saving(clip_state_dict)
        unet_sd = self.diffusion_model.state_dict()
        unet_state_dict = {}
        for k in unet_sd:
            unet_state_dict[k] = ldm.model_management.resolve_lowvram_weight(unet_sd[k], self.diffusion_model, k)

        unet_state_dict = self.model_config.process_unet_state_dict_for_saving(unet_state_dict)
        vae_state_dict = self.model_config.process_vae_state_dict_for_saving(vae_state_dict)
        if self.get_dtype() == torch.float16:
            clip_state_dict = utils.convert_sd_to(clip_state_dict, torch.float16)
            vae_state_dict = utils.convert_sd_to(vae_state_dict, torch.float16)

        if self.model_type == ModelType.V_PREDICTION:
            unet_state_dict["v_pred"] = torch.tensor([])

        return {**unet_state_dict, **vae_state_dict, **clip_state_dict}

    def set_inpaint(self):
        self.inpaint_model = True

    def memory_required(self, input_shape):
        if ldm.model_management.xformers_enabled() or ldm.model_management.pytorch_attention_flash_attention():
            # TODO: this needs to be tweaked
            area = input_shape[0] * input_shape[2] * input_shape[3]
            return (area * ldm.model_management.dtype_size(self.get_dtype()) / 50) * (1024 * 1024)
        else:
            # TODO: this formula might be too aggressive since I tweaked the sub-quad and split algorithms to use less memory.
            area = input_shape[0] * input_shape[2] * input_shape[3]
            return (((area * 0.6) / 0.9) + 1024) * (1024 * 1024)

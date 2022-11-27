from transformers.models.xlm_roberta.configuration_xlm_roberta import XLMRobertaConfig
from transformers.models.clip.configuration_clip import CLIPConfig


class RobertaSeriesConfig(XLMRobertaConfig):
    def __init__(self, pad_token_id=1, bos_token_id=0, eos_token_id=2,project_dim=512,pooler_fn='cls', **kwargs):
        super().__init__(pad_token_id, bos_token_id, eos_token_id, **kwargs)
        self.project_dim = project_dim
        self.pooler_fn = pooler_fn

class AltCLIPConfig(CLIPConfig):
    def __init__(self, text_model_name=None,vision_model_name=None,text_config_dict=None, vision_config_dict=None, projection_dim=512, logit_scale_init_value=2.6592, **kwargs):
        super().__init__(text_config_dict, vision_config_dict, projection_dim, logit_scale_init_value, **kwargs)
        if text_config_dict is None:
            text_config_dict = {}
        # when reload the config from local, we need name to select which class should be instanced.
        self.text_config = RobertaSeriesConfig(**kwargs.pop('text_config'))
        self.text_model_name = text_model_name
        self.vision_model_name = vision_model_name
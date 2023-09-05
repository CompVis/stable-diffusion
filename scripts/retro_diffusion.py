from typing import Dict
import sdkit
from rich import print as rprint

class RetroDiffusion:
    def __init__(self):
        self.logger = rprint
        self.context = sdkit.Context()
        self.loras: Dict[str, float] = {}
        self.lora_alpha: float = 0.0 # TODO: Remove this when we support multiple loras
        self.sounds: bool = False
        
rd = RetroDiffusion()
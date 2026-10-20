from controlvae_plugin.settings import *

@attr.s(auto_attribs=True)
class ControlVAEMuscleSettings(ControlVAESettings):
    controlvae_weight_muscle_length: float = 0.1
    world_model_weight_muscle_length: float = 1
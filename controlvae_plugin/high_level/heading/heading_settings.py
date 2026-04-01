from controlvae_plugin.settings import *

@attr.s(auto_attribs=True)
class ControlVAEHeadingSetttings(ControlVAESettings):
    heading_lr: float = 0.001
    heading_lr_multiplier: float = 0.99

    weight_dir: float = 1
    weight_speed: float = 0
    weight_fall_down: float = 100
    weight_acs: float = 20

    high_level_batch_sz: int = 512
    high_level_rollout_len: int = 16
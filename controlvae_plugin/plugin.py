from typing import Any, Dict, Tuple



def get_trainer_and_settings() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    from controlvae_plugin.trainer import ControlVAETrainer
    from controlvae_plugin.settings import ControlVAESettings
    
    from controlvae_plugin.high_level.heading.heading_trainer import ControlVAEHeadingTrainer
    from controlvae_plugin.high_level.heading.heading_settings import ControlVAEHeadingSetttings

    trainer_types = {
        "ControlVAE": ControlVAETrainer,
        "ControlVAE-Heading": ControlVAEHeadingTrainer,
    }

    trainer_settings = {
        "ControlVAE": ControlVAESettings,
        "ControlVAE-Heading": ControlVAEHeadingSetttings,
    }

    return trainer_types, trainer_settings


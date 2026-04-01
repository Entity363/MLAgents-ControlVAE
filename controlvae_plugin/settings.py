from typing import Dict, Optional, Tuple, List
import os
import torch
from ControlVAECore.Utils.motion_utils import state2ob
from mlagents.torch_utils import torch, default_device
import numpy as np
from collections import defaultdict

from mlagents.trainers.buffer import AgentBuffer, AgentBufferField
from mlagents.trainers.trajectory import ObsUtil
from mlagents.trainers.torch_entities.components.bc.module import BCModule
from mlagents.trainers.torch_entities.components.reward_providers import (
    create_reward_provider,
)

from mlagents.trainers.policy.torch_policy import TorchPolicy
from mlagents.trainers.optimizer import Optimizer
from mlagents.trainers.settings import (
    TrainerSettings,
    RewardSignalSettings,
    RewardSignalType,
)
from mlagents.trainers.torch_entities.utils import ModelUtils
from mlagents.trainers.settings import OnPolicyHyperparamSettings

import random
from typing import List,Dict, cast
import torch.distributions as D
from ControlVAECore.Model.trajectory_collection import TrajectorCollector
from ControlVAECore.Model.world_model import SimpleWorldModel
from ControlVAECore.Utils.mpi_utils import gather_dict_ndarray
from ControlVAECore.Utils.replay_buffer import ReplayBuffer
from tensorboardX import SummaryWriter
from ControlVAECore.Model.modules import *
from ControlVAECore.Utils.motion_utils import *
from ControlVAECore.Utils import pytorch_utils as ptu
import time
import sys
from ControlVAECore.Utils.radam import RAdam

import attr
from enum import Enum

class ControlVAEMode(Enum):
    INFERENCE: str = "Inference"
    TRAINING: str = "Training"

class ControlVAEType(Enum):
    TRACKING: str = "Tracking"
    PRIOR: str = "Prior"
    HEADING: str = "Heading"

@attr.s(auto_attribs=True)
class ControlVAESettings(OnPolicyHyperparamSettings):
    mode: ControlVAEMode = ControlVAEMode.TRAINING
    type: ControlVAEType = ControlVAEType.TRACKING

    with_noise: bool = True
    action_sigma: float = 0.05
    replay_buffer_size: int = 50000
    sub_iter: int = 8

    encoder_hidden_size: int = 512
    encoder_hidden_layer_num : int = 2
    encoder_activation : str = "ELU"
    encoder_fix_var : float = 0.3

    actor_hidden_size: int = 512
    actor_hidden_layer_num : int = 3
    actor_activation : str = "ELU"
    actor_num_experts : int = 6
    actor_gate_hidden_size : int = 64

    latent_size: int = 64

    weight_l2: float = 0.001
    weight_l1: float = 0.01
    weight_kl : float = 0.1

    policy_rollout_length: int = 24
    controlvae_batch_size: int = 512
    controlvae_lr: float = 0.00001
    controlvae_weight_avel : float = 0.5
    controlvae_weight_height : float = 1.2
    controlvae_weight_pos : float = 0.2
    controlvae_weight_rot : float = 0.1
    controlvae_weight_up_dir : float = 3
    controlvae_weight_vel : float = 0.5

    world_model_rollout_length : int = 8
    world_model_lr: float = 0.002
    world_model_activation : str = "ELU"
    world_model_batch_size : int = 512
    world_model_hidden_layer_num : int = 4
    world_model_hidden_layer_size : int = 512

    world_model_weight_pos : float = 1
    world_model_weight_rot : float = 1
    world_model_weight_vel : float = 4
    world_model_weight_avel : float = 4
    #implement others


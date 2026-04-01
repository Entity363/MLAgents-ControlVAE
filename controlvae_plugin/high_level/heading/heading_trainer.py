import abc
from collections import defaultdict
import copy
from typing import cast, Type, Union, Dict, Any

import numpy as np

from mlagents.torch_utils import torch, default_device
from mlagents_envs.base_env import BehaviorSpec
from mlagents_envs.logging_util import get_logger
from mlagents.trainers.buffer import BufferKey, RewardSignalUtil
from mlagents.trainers.trainer.on_policy_trainer import OnPolicyTrainer, RLTrainer
from mlagents.trainers.policy.policy import Policy
from mlagents.trainers.trainer.trainer_utils import get_gae
from mlagents.trainers.optimizer.torch_optimizer import TorchOptimizer
from mlagents.trainers.policy.torch_policy import TorchPolicy
from mlagents.trainers.ppo.optimizer_torch import TorchPPOOptimizer, PPOSettings
from mlagents.trainers.trajectory import Trajectory
from mlagents.trainers.behavior_id_utils import BehaviorIdentifiers
from mlagents.trainers.settings import TrainerSettings
from mlagents.trainers.model_saver.model_saver import BaseModelSaver
from mlagents.trainers.torch_entities.networks import SimpleActor, SharedActorCritic, ObsUtil

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.side_channel import (
    SideChannel,
    IncomingMessage,
    OutgoingMessage,
)
import numpy as np
import uuid

import random
from typing import List,Dict
from ControlVAECore.Model.trajectory_collection import TrajectorCollector
from ControlVAECore.Model.world_model import SimpleWorldModel
from ControlVAECore.Utils.mpi_utils import gather_dict_ndarray
from ControlVAECore.Utils.replay_buffer import ReplayBuffer
from tensorboardX import SummaryWriter
from ControlVAECore.Model.modules import *
from ControlVAECore.Utils.motion_utils import *
from ControlVAECore.Utils.diff_quat import *
from ControlVAECore.Utils import pytorch_utils as ptu
import time
import sys
from ControlVAECore.Utils.radam import RAdam
from mpi4py import MPI

from controlvae_plugin.high_level.heading.heading_actor import ControlVAEHeadingActor
#from .heading_actor import ControlVAEHeadingActor
from controlvae_plugin.policy import ControlVAEPolicy
from controlvae_plugin.high_level.heading.heading_optimizer import ControlVAEHeadingOptimizer
from controlvae_plugin.high_level.heading.heading_settings import *
from controlvae_plugin.side_channel import CustomSideChannel
from controlvae_plugin.saver import ControlVAEModelSaver
from controlvae_plugin.trainer import ControlVAETrainer
from controlvae_plugin.debug.debug import dump_actor, watch_param


logger = get_logger(__name__)

TRAINER_NAME = "ControlVAE-Heading"

class ControlVAEHeadingTrainer(ControlVAETrainer):
    def __init__(
        self,
        behavior_name: str,
        reward_buff_cap: int,
        trainer_settings: TrainerSettings,
        training: bool,
        load: bool,
        seed: int,
        artifact_path: str,
    ):
        super().__init__(
            behavior_name,
            reward_buff_cap,
            trainer_settings,
            training,
            load,
            seed,
            artifact_path,
        )
        self.trainer_settings = trainer_settings

        self.replay_buffer.reset_max_size(2000)

        #implement in optimizer and pass in here
        self.hyperparameters: ControlVAEHeadingSetttings = cast(
            ControlVAEHeadingSetttings, self.trainer_settings.hyperparameters
        )

    #need to reinitialize to remove target and add heading
    @property
    def replay_buffer_keys(self):
        return ['state', 'action', 'heading']

    @staticmethod
    def get_trainer_name() -> str:
        return TRAINER_NAME
    
    def create_optimizer(self) -> TorchOptimizer:
        self.optimizer = ControlVAEHeadingOptimizer(  # type: ignore
            cast(ControlVAEPolicy, self.policy), self.trainer_settings,  # type: ignore
            statics= self.statistics
        )  # type: ignore
        return self.optimizer
    
    def create_actor(self):
        actor = ControlVAEHeadingActor
        return actor
    
    @property
    def high_level_data_name_list(self):
        return ['state', 'heading']
    

    
    def _update_policy(self):
        name_list = self.high_level_data_name_list
        rollout_length = self.hyperparameters.high_level_rollout_len
        batch_sz = self.hyperparameters.high_level_batch_sz
        sub_iter = self.hyperparameters.sub_iter
        data_loader = self.replay_buffer.\
            generate_data_loader(   name_list,
                                    rollout_length,
                                    batch_sz,
                                    sub_iter
                                )
        for batch_dict in data_loader:
            #print("training high level")
            log = self.optimizer.train_high_level(*batch_dict)
            #print("trained high level")
        self.optimizer.scheduler.step()
        #print("[TRAINER] Stepped optimizer scheduler")
        logger.info(log)

        time3 = time.perf_counter()
        #time to go trough the whole process
        if self._last_iteration_end is not None:
            log['iteration_time'] = time3 - self._last_iteration_end
        self._last_iteration_end = time3
        #print("[TRAINER] timed")

        for key, value in log.items():
            if torch.is_tensor(value):
                value = value.detach().mean().item()
            elif isinstance(value, np.ndarray):
                value = float(np.mean(value))
            else:
                value = float(value)

            self.stats_reporter.add_stat(key, value)
        #print("[TRAINER] Added stats")
        self.policy.count = 0

        return True
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

from controlvae_plugin.high_level.muscles.muscle_optimizer import ControlVAEMuscleOptimizer
from controlvae_plugin.high_level.muscles.muscle_settings import ControlVAEMuscleSettings
from controlvae_plugin.high_level.muscles.muscle_actor import ControlVAEMuscleActor


logger = get_logger(__name__)

TRAINER_NAME = "ControlVAE-Muscle"

class ControlVAEMuscleTrainer(ControlVAETrainer):
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

        #implement in optimizer and pass in here
        self.hyperparameters: ControlVAEMuscleSettings = cast(
            ControlVAEMuscleSettings, self.trainer_settings.hyperparameters
        )

        self.actor_kwargs["world_model_weight_muscle_length"] = self.hyperparameters.world_model_weight_muscle_length


    #need to reinitialize to remove target and add heading
    @property
    def replay_buffer_keys(self):
        return ['state', 'action', 'target', 'target_muscle_lengths', 'state_muscle_lengths']

    @staticmethod
    def get_trainer_name() -> str:
        return TRAINER_NAME
    
    def create_optimizer(self) -> TorchOptimizer:
        self.optimizer = ControlVAEMuscleOptimizer(  # type: ignore
            cast(ControlVAEPolicy, self.policy), self.trainer_settings,  # type: ignore
            statics= self.statistics
        )  # type: ignore
        return self.optimizer
    
    def create_actor(self):
        actor = ControlVAEMuscleActor
        return actor
    
    @property
    def world_model_data_name(self):
        return ['state', 'action'] #, 'state_muscle_lengths'
    
    @property
    def policy_data_name(self):
        return ['state', 'state_muscle_lengths', 'target', 'target_muscle_lengths']
        
        
    def _update_policy(self):
        time1 = time.perf_counter()
        # for batch in data loader train world model, policy
        name_list = self.world_model_data_name
        rollout_length = self.hyperparameters.world_model_rollout_length
        wm_batch_size = self.hyperparameters.world_model_batch_size
        sub_iter = self.hyperparameters.sub_iter
        data_loader = self.replay_buffer.generate_data_loader(name_list,
                                                rollout_length+1,
                                                wm_batch_size,
                                                sub_iter
        )
        for batch in  data_loader:
            world_model_log = self.optimizer.train_world_model(*batch)

        time2 = time.perf_counter()

        name_list = self.policy_data_name
        rollout_length = self.hyperparameters.policy_rollout_length
        policy_batch_size = self.hyperparameters.controlvae_batch_size
        data_loader = self.replay_buffer.generate_data_loader(name_list,
                                                rollout_length,
                                                policy_batch_size,
                                                sub_iter
        )
        for batch in  data_loader:
            policy_log = self.optimizer.train_policy(*batch)

        time3 = time.perf_counter()

        world_model_log['training_time'] = (time2-time1)
        policy_log['training_time'] = (time3-time2)

        merged_log = self.merge_dict([world_model_log, policy_log], ['world_model', 'policy'])

        #time to go trough the whole process
        if self._last_iteration_end is not None:
            merged_log['iteration_time'] = time3 - self._last_iteration_end
            self._last_iteration_end = time3


        logger.info(merged_log)

        for key, value in merged_log.items():
            if torch.is_tensor(value):
                value = value.detach().mean().item()
            elif isinstance(value, np.ndarray):
                value = float(np.mean(value))
            else:
                value = float(value)

            self.stats_reporter.add_stat(key, value)

        self.policy.count = 0

        return True
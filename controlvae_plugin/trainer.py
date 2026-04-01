
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


from .actor import ControlVAEActor
from .policy import ControlVAEPolicy
from .optimizer import ControlVAEOptimizer
from .settings import *
from .side_channel import CustomSideChannel
from .saver import ControlVAEModelSaver
import shared_statics
from .debug.debug import dump_actor, watch_param


logger = get_logger(__name__)

TRAINER_NAME = "ControlVAE"

class ControlVAETrainer(RLTrainer):
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
            trainer_settings,
            training,
            load,
            artifact_path,
            reward_buff_cap
        )
        self.trainer_settings = trainer_settings

        #implement in optimizer and pass in here
        self.hyperparameters: ControlVAESettings = cast(
            ControlVAESettings, self.trainer_settings.hyperparameters
        )
        
        self.seed = seed
        #self.shared_critic = self.hyperparameters.shared_critic
        
        self.policy: ControlVAEPolicy = None  # type: ignore

        self.replay_buffer = ReplayBuffer(self.replay_buffer_keys, self.hyperparameters.replay_buffer_size) # if mpi_rank ==0 else None

        self.statistics = dict(shared_statics.STATICS)

        self.is_ready = False

        self._last_iteration_end = 0
    
    @staticmethod
    def merge_dict(dict_list: List[dict], prefix: List[str]):
        """Merge dict with prefix, used in merge logs from different model

        Args:
            dict_list (List[dict]): different logs
            prefix (List[str]): prefix you hope to add before keys
        """
        res = {}
        for dic, prefix in zip(dict_list, prefix):
            for key, value in dic.items():
                res[prefix+'_'+key] = value
        return res
    
    
    def unpack_raw_np(self, raw: np.ndarray, state_sz, obs_sz) -> dict:
        """
        raw: [B, obs_dim] from Unity (vector obs)
        packed as [state | state_obs | target | target_obs | done | prior_vs_post]
        """

        if raw.ndim == 1:
            raw = raw[None, :]

        N = state_sz
        M = obs_sz

        state       = raw[:, 0:N]
        state_obs   = raw[:, N:N+M]
        target      = raw[:, N+M:N+M+N]
        target_obs  = raw[:, N+M+N:N+M+N+M]
        done        = raw[:, N+M+N+M:N+M+N+M+1]
        prior_vs_post = raw[:, N+M+N+M+1:N+M+N+M+2]

        return {
            "state": state,
            "state_obs": state_obs,
            "target": target,
            "target_obs": target_obs,
            "done": done,
            "prior_vs_post": prior_vs_post,
        }
    
    @property
    def replay_buffer_keys(self):
        return ['state', 'action', 'target']
    
    @property
    def world_model_data_name(self):
        return ['state', 'action']

    @property
    def policy_data_name(self):
        return ['state', 'target']
    
    def _process_trajectory(self, trajectory: Trajectory) -> None:
        """
        Takes a trajectory and processes it, putting it into the update buffer.
        Processing involves calculating value and advantage targets for model updating step.
        :param trajectory: The Trajectory tuple containing the steps to be processed.
        """
        super()._process_trajectory(trajectory)
        agent_id = trajectory.agent_id  # All the agents should have the same ID

        agent_buffer_trajectory = trajectory.to_agentbuffer()

        n_obs = len(self.policy.behavior_spec.observation_specs)
        observations = ObsUtil.from_buffer(agent_buffer_trajectory, n_obs)

        # Stack because it's a list not an np 2d array
        raw_vec = np.stack(observations[0], axis=0).astype(np.float32)

        #sizing
        self.observations_size_vector = raw_vec.shape[1]
        #obs is state sz + normalized state obs sz + target sz + normalized target obs sz + 1 done + 1 prior vs post bool
        """
        self.state_obs_sz = (self.observations_size_vector - 2) // 2
        self.body_sz = (self.state_obs_sz - 3) // 29  #13 body + 16 body + 3
        self.state_sz = self.body_sz * 13
        self.obs_sz = self.body_sz * 16 + 3

        self.target_sz = self.state_sz
        self.target_obs_sz = self.obs_sz
        """
        self.state_sz = self.policy.actor.state_sz
        self.obs_sz = self.policy.actor.obs_sz
        self.body_sz = self.policy.actor.body_sz

        #actual trajectorization
        traj = self.policy.actor.unpack_raw(raw_vec, self.state_sz, self.obs_sz)

        #reshaped to [B, num bodies, 13]
        traj["state"] = traj["state"].reshape(traj["state"].shape[0], self.body_sz, 13)
        #already normalized
        traj["n_observation"] = traj["state_obs"] 

        #need to create target obs from target state here because the unity ones are already normalized, and train policy doesn't want it
        target_state = torch.tensor(
            traj["target"].reshape(traj["target"].shape[0], self.body_sz, 13),
            dtype=torch.float32,
        )
        target_obs = state2ob(target_state).cpu().numpy()
        traj["target"] = target_obs

        #actions
        action = np.stack(
            agent_buffer_trajectory[BufferKey.CONTINUOUS_ACTION],
            axis=0
        ).astype(np.float32)
        traj["action"] = action

        #force last step to be terminal
        traj["done"][-1] = 1.0

        #remove them to save space since they're not needed
        del traj["target_obs"]
        del traj["state_obs"]
        del traj["prior_vs_post"]

        self.replay_buffer.add_trajectory(traj)

        frames = traj["state"].shape[0]
        print(f"\nReceived trajectory: {frames} frames. Worker id: {trajectory.agent_id}\n")

        if self.hyperparameters.mode == ControlVAEMode.TRAINING:
            self.is_ready = True
        elif self.hyperparameters.mode == ControlVAEMode.INFERENCE:
            self.is_ready = False

    def _is_ready_update(self):
        """
        Returns whether or not the trainer has enough elements to run update model
        :return: A boolean corresponding to whether or not update_model() can be run
        """
        if (self.is_ready == True):
            self.is_ready = False
            return True
        else:
            return False

    def create_optimizer(self) -> TorchOptimizer:
        self.optimizer = ControlVAEOptimizer(  # type: ignore
            cast(ControlVAEPolicy, self.policy), self.trainer_settings,  # type: ignore
            statics= self.statistics
        )  # type: ignore
        return self.optimizer

    def create_actor(self):
        actor = ControlVAEActor
        return actor
    
    def create_policy(
        self, parsed_behavior_id: BehaviorIdentifiers, behavior_spec: BehaviorSpec
    ) -> ControlVAEPolicy:
        actor_cls = self.create_actor()

        self.actor_kwargs: Dict[str, Any] = {
            "mode": self.hyperparameters.mode,
            "type": self.hyperparameters.type,

            "action_sigma": self.hyperparameters.action_sigma,
            "with_noise": self.hyperparameters.with_noise,

            "latent_size": self.hyperparameters.latent_size,
            "encoder_fix_var": self.hyperparameters.encoder_fix_var,
            "encoder_hidden_layer_num" : self.hyperparameters.encoder_hidden_layer_num,
            "encoder_hidden_layer_size" : self.hyperparameters.encoder_hidden_size,
            "encoder_activation" : self.hyperparameters.encoder_activation,

            "actor_hidden_layer_size" : self.hyperparameters.actor_hidden_size,
            "actor_hidden_layer_num" : self.hyperparameters.actor_hidden_layer_num,
            "actor_activation" : self.hyperparameters.actor_activation,
            "actor_num_experts" : self.hyperparameters.actor_num_experts,
            "actor_gate_hidden_layer_size" : self.hyperparameters.actor_gate_hidden_size,

            "world_model_hidden_layer_num" : self.hyperparameters.world_model_hidden_layer_num,
            "world_model_hidden_layer_size" : self.hyperparameters.world_model_hidden_layer_size,
            "world_model_activation" : self.hyperparameters.world_model_activation,
            "world_model_weight_pos" : self.hyperparameters.world_model_weight_pos,
            "world_model_weight_vel" : self.hyperparameters.world_model_weight_vel,
            "world_model_weight_rot" : self.hyperparameters.world_model_weight_rot,
            "world_model_weight_avel" : self.hyperparameters.world_model_weight_avel,
        }

        policy = ControlVAEPolicy(
            self.seed,
            behavior_spec,
            self.trainer_settings.network_settings,
            actor_cls,
            self.statistics,
            self.actor_kwargs
        )

        return policy
    
    def get_policy(self, name_behavior_id: str) -> Policy:
        """
        Gets policy from trainer associated with name_behavior_id
        :param name_behavior_id: full identifier of policy
        """

        return self.policy
    

    def add_policy(
        self, parsed_behavior_id: BehaviorIdentifiers, policy: Policy
    ) -> None:
        """
        Adds policy to trainer.
        :param parsed_behavior_id: Behavior identifiers that the policy should belong to.
        :param policy: Policy to associate with name_behavior_id.
        """
        if self.policy:
            logger.warning(
                "Your environment contains multiple teams, but {} doesn't support adversarial games. Enable self-play to \
                    train adversarial games.".format(
                    self.__class__.__name__
                )
            )
        self.policy = policy
        self.policies[parsed_behavior_id.behavior_id] = policy

        self.optimizer = self.create_optimizer()
        for _reward_signal in self.optimizer.reward_signals.keys():
            self.collected_rewards[_reward_signal] = defaultdict(lambda: 0)

        self.model_saver.register(self.policy)
        self.model_saver.register(self.optimizer)
        self.model_saver.initialize_or_load()

        # Needed to resume loads properly
        self._step = policy.get_current_step()
    
    @staticmethod
    def get_trainer_name() -> str:
        return TRAINER_NAME
    

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
    
    
    @staticmethod
    def create_model_saver(trainer_settings: TrainerSettings, model_path: str, load: bool) -> BaseModelSaver:
        model_saver = ControlVAEModelSaver(trainer_settings, model_path, load)
        return model_saver
    

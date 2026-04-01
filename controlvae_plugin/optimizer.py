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

from .debug.debug import dump_actor

from .settings import ControlVAESettings


class ControlVAEOptimizer(Optimizer):
    def __init__(self, policy: TorchPolicy, trainer_settings: TrainerSettings, statics = None):
        self.policy = policy
        self.trainer_settings = trainer_settings

        self.hyperparameters: ControlVAESettings = cast(
            ControlVAESettings, trainer_settings.hyperparameters
        )

        #statics
        self._statics_proxy = statics
        self.statics = dict(self._statics_proxy) if self._statics_proxy is not None else None

        self.obs_mean = torch.as_tensor(
            statics['obs_mean'], dtype=torch.float32, device=default_device()
        )
        self.obs_std = torch.as_tensor(
            statics['obs_std'], dtype=torch.float32, device=default_device()
        )

        self.delta_mean = torch.as_tensor(
            statics['delta_mean'], dtype=torch.float32, device=default_device()
        )
        self.delta_std = torch.as_tensor(
            statics['delta_std'], dtype=torch.float32, device=default_device()
        )
        #print("\noptimizer obs std: \n", self.obs_std)

        # optimizer
        self.wm_optimizer = RAdam(self.policy.actor.world_model.parameters(), self.hyperparameters.world_model_lr, weight_decay=1e-3)
        self.vae_optimizer = RAdam( list(self.policy.actor.encoder.parameters()) + list(self.policy.actor.agent.parameters()), self.hyperparameters.controlvae_lr)
        self.beta_scheduler = ptu.scheduler(0,8,0.009,0.09,500*8)

        self.weight = {
            "pos": self.hyperparameters.controlvae_weight_pos,
            "rot": self.hyperparameters.controlvae_weight_rot,
            "vel": self.hyperparameters.controlvae_weight_vel,
            "avel": self.hyperparameters.controlvae_weight_avel,
            "height": self.hyperparameters.controlvae_weight_height,
            "up_dir": self.hyperparameters.controlvae_weight_up_dir,
            "l2": self.hyperparameters.weight_l2,
            "l1": self.hyperparameters.weight_l1,
            "kl": self.hyperparameters.weight_kl,
        }

        self.reward_signals = {}

    def _ensure_statics(self):
        return self._refresh_stats_tensors()

    def _refresh_stats_tensors(self):
        if self._statics_proxy is None or len(self._statics_proxy) == 0:
            return False

        self.statics = dict(self._statics_proxy)

        self.obs_mean = torch.as_tensor(
            self.statics["obs_mean"], dtype=torch.float32, device=default_device()
        )
        self.obs_std = torch.as_tensor(
            self.statics["obs_std"], dtype=torch.float32, device=default_device()
        )
        self.delta_mean = torch.as_tensor(
            self.statics["delta_mean"], dtype=torch.float32, device=default_device()
        )
        self.delta_std = torch.as_tensor(
            self.statics["delta_std"], dtype=torch.float32, device=default_device()
        )
        return True
    
    def update(self, batch: AgentBuffer, num_sequences: int) -> Dict[str, float]:
        if not self._ensure_statics():
            return
        
    def normalize_obs(self, observation):
        if isinstance(observation, np.ndarray):
            observation = ptu.from_numpy(observation)
        if len(observation.shape) == 1:
            observation = observation[None, ...]
        nobs = ptu.normalize(
            observation,
            self.obs_mean,
            self.obs_std
        )
        return nobs
    
    def obsinfo2n_obs(self, obs_info):
        if 'n_observation' in obs_info:
            n_observation = obs_info['n_observation']
        else:
            if 'observation' in obs_info:
                observation = obs_info['observation']
            else:
                observation = state2ob(obs_info['state'])
            n_observation = self.normalize_obs(observation)
        return n_observation
    
    def grad_norm(self, module):
        vals = []
        for p in module.parameters():
            if p.grad is not None:
                vals.append(p.grad.abs().mean().item())
        return 0.0 if not vals else sum(vals) / len(vals)

    def act_tracking(self, **obs_info):
        target = obs_info["target"]

        n_target = self.normalize_obs(target)
        n_observation = self.obsinfo2n_obs(obs_info)

        latent_code, mu_post, mu_prior = self.encode(n_observation, n_target)
        action = self.decode(n_observation, latent_code)
        info = {
            "mu_prior": mu_prior,
            "mu_post": mu_post
        }
        return action, info
    
    def encode(self, normalized_obs, normalized_target, **kargs):
        """encode observation and target into posterior distribution

        Args:
            normalized_obs (Optional[Tensor,np.ndarray]): normalized current observation
            normalized_target (Optional[Tensor, np.ndarray]): normalized current target 

        Returns:
            Tuple(tensor, tensor, tensor): 
                latent coder, mean of prior distribution, mean of posterior distribution 
        """
        return self.policy.actor.encoder(normalized_obs, normalized_target)
    
    def decode(self, normalized_obs, latent, **kargs):
        """decode latent code into action space

        Args:
            normalized_obs (tensor): normalized current observation
            latent (tensor): latent code

        Returns:
            tensor: action
        """
        action = self.policy.actor.agent(latent, normalized_obs)        
        return action

    def train_policy(self, states, targets):
        if not self._ensure_statics():
            return {}
        
        rollout_length = states.shape[1]
        loss_name = ['pos', 'rot', 'vel', 'avel', 'height', 'up_dir', 'acs', 'kl']
        loss_num = len(loss_name)
        loss = list( ([] for _ in range(loss_num)) )
        states = states.transpose(0,1).contiguous().to(default_device())
        targets = targets.transpose(0,1).contiguous().to(default_device())
        cur_state = states[0]
        cur_observation = state2ob(cur_state)
        n_observation = self.normalize_obs(cur_observation)
        for i in range(rollout_length):
            target = targets[i]
            action, info = self.act_tracking(n_observation = n_observation, target = target)
            action = action + torch.randn_like(action)*0.05
            cur_state = self.policy.actor.world_model(cur_state, action, n_observation = n_observation)
            cur_observation = state2ob(cur_state)
            """ 
            stastics come from the sidechannel
            """
            n_observation = self.normalize_obs(cur_observation)

            loss_tmp = pose_err(cur_observation, target, self.weight, dt = 0.02)
            for j, value in enumerate(loss_tmp):
                loss[j].append(value)        
            acs_loss = self.hyperparameters.weight_l2 * torch.mean(torch.sum(action**2,dim = -1)) \
                + self.hyperparameters.weight_l1 * torch.mean(torch.norm(action, p=1, dim=-1))
            kl_loss = self.policy.actor.encoder.kl_loss(**info)
            kl_loss = torch.mean( torch.sum(kl_loss, dim = -1))
            loss[-2].append(acs_loss)
            loss[-1].append(kl_loss * self.beta_scheduler.value)
        
        loss_value = [ sum( (0.95**i)*l[i] for i in range(rollout_length) )/rollout_length for l in loss]
        loss = sum(loss_value)

        self.vae_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.actor.encoder.parameters(), 1, error_if_nonfinite=True)
        torch.nn.utils.clip_grad_norm_(self.policy.actor.parameters(), 1, error_if_nonfinite= True)

        self.vae_optimizer.step()
        self.beta_scheduler.step()

        res = {loss_name[i]: loss_value[i] for i in range(loss_num)}
        res['beta'] = self.beta_scheduler.value
        res['loss'] = loss
        return res
    
    
    def train_world_model(self, states, actions):
        if not self._ensure_statics():
            return
        
        rollout_length = states.shape[1] -1
        loss_name = ['pos', 'rot', 'vel', 'avel']
        loss_num = len(loss_name)
        loss = list( ([] for _ in range(loss_num)) )
        states = states.transpose(0,1).contiguous().to(default_device())
        actions = actions.transpose(0,1).contiguous().to(default_device())
        cur_state = states[0]
        for i in range(rollout_length):
            next_state = states[i+1]
            pred_next_state = self.policy.actor.world_model(cur_state, actions[i])
            loss_tmp = self.policy.actor.world_model.loss(pred_next_state, next_state)
            cur_state = pred_next_state
            for j in range(loss_num):
                loss[j].append(loss_tmp[j])
        
        loss_value = [sum(i) for i in loss]
        loss = sum(loss_value)
        
        self.wm_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.actor.world_model.parameters(), 1, error_if_nonfinite=True)

        self.wm_optimizer.step()

        res= {loss_name[i]: loss_value[i] for i in range(loss_num)}
        res['loss'] = loss
        return res
    
    
    def get_modules(self):
        modules = {
            "Optimizer:vae_optimizer": self.vae_optimizer,
            "Optimizer:wm_optimizer": self.wm_optimizer,
        }
        for reward_provider in self.reward_signals.values():
            modules.update(reward_provider.get_modules())
        return modules
    
    @property
    def critic(self):
        pass
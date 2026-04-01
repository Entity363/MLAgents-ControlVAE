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

from ...debug.debug import dump_actor

from ...settings import ControlVAESettings
from ...optimizer import ControlVAEOptimizer
from PlayGround.playground_util import *

from .heading_settings import ControlVAEHeadingSetttings


class ControlVAEHeadingOptimizer(ControlVAEOptimizer):
    def __init__(self, policy: TorchPolicy, trainer_settings: TrainerSettings, statics = None):
        super().__init__(policy, trainer_settings, statics)

        self.hyperparameters: ControlVAEHeadingSetttings = cast(
            ControlVAEHeadingSetttings, trainer_settings.hyperparameters
        )

        self.weight = {
            "direction": self.hyperparameters.weight_dir,
            "speed": self.hyperparameters.weight_speed,
            "fall_down": self.hyperparameters.weight_fall_down,
            "acs": self.hyperparameters.weight_acs
        }
        
        self.high_level_optim = RAdam(self.policy.actor.high_level.parameters(), lr=self.hyperparameters.heading_lr)
        lr = lambda epoach: max(self.hyperparameters.heading_lr_multiplier**(epoach), 1e-1)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.high_level_optim, lr)

    def get_modules(self):
        modules = {
            "Optimizer:vae_optimizer": self.vae_optimizer,
            "Optimizer:wm_optimizer": self.wm_optimizer,
            "Optimizer:heading_optimizer": self.high_level_optim
        }
        for reward_provider in self.reward_signals.values():
            modules.update(reward_provider.get_modules())
        return modules
    
    
    def get_loss(self, state, target):
        #print("[OPTIMIZER] get loss intro")
        direction = get_root_facing(state)
        #print("[OPTIMIZER] get loss after get root facing")
        delta_angle = torch.atan2(direction[:,2], direction[:,0]) - target[:,1]
        direction_loss = torch.acos(torch.cos(delta_angle).clamp(min=-1+1e-4, max=1-1e-4))/ torch.pi
        #print("[OPTIMIZER] get loss after dir loss")

        """
        com_vel = state2speed(state, self.mass)
        #print("[OPTIMIZER] get loss after state2speed")
        target_direction = torch.cat([torch.cos(target[:,1,None]),torch.sin(target[:,1,None])], dim=-1)
        #print("[OPTIMIZER] get loss before einsum")
        com_vel = torch.where(target[:,0]==0, torch.norm(com_vel, dim=-1, p = 1), torch.einsum('bi,bi->b', com_vel[:,[0,2]], target_direction))
        # com_vel = torch.einsum('bi,bi->b', com_vel[:,[0,2]], target_direction)
        # com_vel = torch.norm(com_vel, dim = -1)
        speed_loss = torch.abs(com_vel - target[:,0])/target[:,0].clamp(min=1)"""
        root_vel = state[:, 0, 7:10]  # pelvis/root linear velocity
        target_direction = torch.cat(
            [torch.cos(target[:, 1, None]), torch.sin(target[:, 1, None])],
            dim=-1,
        )
        speed_along_target = torch.where(
            target[:, 0] == 0,
            torch.norm(root_vel[:, [0, 2]], dim=-1, p=1),
            torch.einsum('bi,bi->b', root_vel[:, [0, 2]], target_direction),
        )
        speed_loss = torch.abs(speed_along_target - target[:, 0]) / target[:, 0].clamp(min=1)
        #print("[OPTIMIZER] get loss after speed loss")
        
        fall_down_loss = torch.clamp(state[...,0,1], min = 0, max = 0.6)
        fall_down_loss = (0.6 - fall_down_loss)
        fall_down_loss = torch.mean(fall_down_loss)
        #print("[OPTIMIZER] get loss after fall down loss")
        
        return direction_loss.mean(), speed_loss.mean(), fall_down_loss

    def act_task(self, **obs_info):
        
        n_observation = self.obsinfo2n_obs(obs_info)
        latent, mu, _ = self.policy.actor.encoder.encode_prior(n_observation)    
        n_target = self.policy.actor.target2n_target(obs_info['state'], obs_info['target'])
        
        task = torch.cat([n_observation, n_target], dim=1)
        offset = self.policy.actor.high_level(task)
        if self.policy.actor.dance:
            if n_target[...,2].abs()<0.5:
                latent = latent
            else:
                latent = latent + offset
        else:
            latent = mu+offset
        
        action = self.decode(n_observation, latent)
        return action, {
            'mu': mu,
            'latent': latent,
            'offset': offset
        }
    
    def train_high_level(self, states, targets):
        rollout_length = states.shape[1]
        cur_state = states[:,0].to(default_device())
        targets = targets.to(default_device())
        cur_observation = state2ob(cur_state)
        n_observation = self.normalize_obs(cur_observation)
        
        loss_name = ['direction', 'speed', 'fall_down', 'acs']
        loss_num = len(self.weight)
        loss = [[] for i in range(loss_num)]
        
        #print("[OPTIMIZER] train high level before loop")

        # speed = np.random.choice(self.env.speed_range, targets[:,0,0].shape)
        # targets[:,:,0] = ptu.from_numpy(speed)[:,None]
        for i in range(rollout_length):
            #synthetic step
            #print("[OPTIMIZER] train high level before act task")
            action, info = self.act_task(state = cur_state, target = targets[:,i], n_observation = n_observation)
            #print("[OPTIMIZER] train high level after act task")
            cur_state = self.policy.actor.world_model(cur_state, action, n_observation = n_observation)
            #print("[OPTIMIZER] train high level after wm")
            cur_observation = state2ob(cur_state)
            n_observation = self.normalize_obs(cur_observation)
            #print("[OPTIMIZER] train high level before get loss")
            # cal_loss
            loss_tmp = self.get_loss(cur_state, targets[:,i])
            for j, value in enumerate(loss_tmp):
                loss[j].append(value)
            action_loss = torch.mean(info['offset']**2)
            loss[-1].append(action_loss)
        
        #print("[OPTIMIZER] train high level after loop")
        #optimizer step
        # weight = [1,1,100,20]
        #weight = [1,0,100,20]
        loss_value = [sum(l)/rollout_length*self.weight[loss_name[i]] for i,l in enumerate(loss)]
        #print("[OPTIMIZER] train high level after loss value")
        loss_value[0] = loss[0][-1]
        loss = sum(loss_value)
        #print("[OPTIMIZER] train high level after loss computation")
        
        
        self.high_level_optim.zero_grad()
        loss.backward()
        #print("[OPTIMIZER] train high level before clip grad norm")
        torch.nn.utils.clip_grad_norm_(self.policy.actor.high_level.parameters(), 1)
        self.high_level_optim.step()

        #print("[OPTIMIZER] train high level after loss")
        
        # return
        res = {loss_name[i]: loss_value[i] for i in range(loss_num)}
        return res
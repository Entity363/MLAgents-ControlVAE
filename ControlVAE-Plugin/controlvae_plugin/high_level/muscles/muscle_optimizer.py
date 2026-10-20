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

from .muscle_settings import ControlVAEMuscleSettings

class ControlVAEMuscleOptimizer(ControlVAEOptimizer):
    def __init__(self, policy: TorchPolicy, trainer_settings: TrainerSettings, statics = None):
        super().__init__(policy, trainer_settings, statics)

        self.hyperparameters: ControlVAEMuscleSettings = cast(
            ControlVAEMuscleSettings, trainer_settings.hyperparameters
        )

        self.body_weights = torch.as_tensor(
            self.statics["body_weights"], dtype=torch.float32, device=default_device()
        )
        # need to normalize muscle lengths by rest length to make the learning easier
        self.muscle_length_scale = torch.as_tensor(
            self.statics["rest_lengths"], dtype=torch.float32, device=default_device()
        )
        self.rest_se_slack_lengths = torch.as_tensor(
            self.statics["rest_se_slack_lengths"], dtype=torch.float32, device=default_device()
        )
        self.rest_optimal_ce_lengths = torch.as_tensor(
            self.statics["rest_optimal_ce_lengths"], dtype=torch.float32, device=default_device()
        )
        # for length recomposition in policy
        self.body_indices= torch.as_tensor(
            self.statics["body_indices"], dtype=torch.float32, device=default_device()
        )
        self.muscle_indices= torch.as_tensor(
            self.statics["muscle_indices"], dtype=torch.float32, device=default_device()
        )
        #offsets vec3s must be sent as separate floats
        self.muscle_attachment_pointsx = torch.as_tensor(
            self.statics["muscle_offset_x"], dtype=torch.float32, device=default_device()
        )
        self.muscle_attachment_pointsy = torch.as_tensor(
            self.statics["muscle_offset_y"], dtype=torch.float32, device=default_device()
        )
        self.muscle_attachment_pointsz = torch.as_tensor(
            self.statics["muscle_offset_z"], dtype=torch.float32, device=default_device()
        )
        self.muscle_attachment_points = torch.stack([self.muscle_attachment_pointsx, self.muscle_attachment_pointsy, self.muscle_attachment_pointsz], dim = -1)


        self.weight = {
            "pos": self.hyperparameters.controlvae_weight_pos,
            "rot": self.hyperparameters.controlvae_weight_rot,
            "vel": self.hyperparameters.controlvae_weight_vel,
            "avel": self.hyperparameters.controlvae_weight_avel,
            "height": self.hyperparameters.controlvae_weight_height,
            "up_dir": self.hyperparameters.controlvae_weight_up_dir,
            "muscle_length": self.hyperparameters.controlvae_weight_muscle_length,
            "l2": self.hyperparameters.weight_l2,
            "l1": self.hyperparameters.weight_l1,
            "kl": self.hyperparameters.weight_kl,
        }

    def act_tracking(self, **obs_info):
        target = obs_info["target"]

        n_target = self.normalize_obs(target)
        n_observation = self.obsinfo2n_obs(obs_info)
        base_muscle_length = obs_info["state_muscle_lengths"]
        target_muscle_length = obs_info["target_muscle_lengths"]

        latent_code, mu_post, mu_prior = self.encode(torch.cat([n_observation, base_muscle_length], dim=-1), torch.cat([n_target, target_muscle_length], dim=-1))
        action = self.decode(torch.cat([n_observation, base_muscle_length], dim=-1), latent_code)
        info = {
            "mu_prior": mu_prior,
            "mu_post": mu_post
        }
        return action, info

    # remember to add muscle lengths to the replay buffer keys
    def train_policy(self, states, state_muscle_lengths, targets, target_muscle_lengths):
        if not self._ensure_statics():
            return {}
        
        rollout_length = states.shape[1]
        loss_name = ['pos', 'rot', 'vel', 'avel', 'height', 'up_dir', 'muscle_length', 'acs', 'kl']
        loss_num = len(loss_name)
        loss = list( ([] for _ in range(loss_num)) )
        states = states.transpose(0,1).contiguous().to(default_device())
        targets = targets.transpose(0,1).contiguous().to(default_device())
        target_muscle_lengths = target_muscle_lengths.transpose(0,1).contiguous().to(default_device())
        state_muscle_lengths = state_muscle_lengths.transpose(0,1).contiguous().to(default_device())
        cur_state = states[0]
        cur_observation = state2ob(cur_state)
        n_observation = self.normalize_obs(cur_observation)

        state_muscle_length = state_muscle_lengths[0]

        for i in range(rollout_length):
            target = targets[i]
            target_muscle_length = target_muscle_lengths[i]
            #state_muscle_length = state_muscle_lengths[i] #get_muscle_lengths(cur_state, self.muscle_attachment_points, self.body_indices, self.muscle_indices) / self.muscle_length_scale
            action, info = self.act_tracking(n_observation = n_observation, target = target, state_muscle_lengths = state_muscle_length, target_muscle_lengths = target_muscle_length)
            action = action + torch.randn_like(action)*0.05
            cur_state = self.policy.actor.world_model(cur_state, action, n_observation = n_observation)
            cur_observation = state2ob(cur_state)
            """ 
            stastics come from the sidechannel
            """
            n_observation = self.normalize_obs(cur_observation)

            #implement
            pos_loss, rot_loss,vel_loss, avel_loss, height_loss, up_dir_loss, = pose_err_with_body_weights(cur_observation, target, self.weight, self.body_weights, dt = 0.02)
            for j, value in enumerate([pos_loss, rot_loss, vel_loss, avel_loss, height_loss, up_dir_loss]):
                loss[j].append(value)

            #check this works or not
            predicted_muscle_length = get_muscle_lengths(cur_state, self.muscle_attachment_points, self.body_indices, self.muscle_indices) 
            # (length/rest - rest se slack) / rest optimal ce
            state_muscle_length = ((predicted_muscle_length / self.muscle_length_scale) - self.rest_se_slack_lengths) / self.rest_optimal_ce_lengths
            
            #print("delta: ", state_muscle_length - target_muscle_length)

            #print(self.hyperparameters.controlvae_weight_muscle_length)

            muscle_lengths_loss = torch.mean(torch.norm(state_muscle_length - target_muscle_length,  p=1, dim=-1)) \
                * self.hyperparameters.controlvae_weight_muscle_length \
                / self.muscle_length_scale.size(0) #normalize by number of muscles
            #muscle_lengths_loss.clamp_(max = 5.0)
            loss[loss_name.index('muscle_length')].append(muscle_lengths_loss)


            acs_loss = self.hyperparameters.weight_l2 * torch.mean(torch.sum(action**2,dim = -1)) \
                + self.hyperparameters.weight_l1 * torch.mean(torch.norm(action, p=1, dim=-1))
            kl_loss = self.policy.actor.encoder.kl_loss(**info)
            kl_loss = torch.mean( torch.sum(kl_loss, dim = -1))
            #kl_loss = self.weight['kl'] * kl_loss
            #kl_loss = torch.clamp(kl_loss, max = 1.0)
            loss[-2].append(acs_loss)
            loss[-1].append(kl_loss * self.beta_scheduler.value * self.weight['kl'])
        
        loss_value = [ sum( (0.95**i)*l[i] for i in range(rollout_length) )/rollout_length for l in loss]
        loss = sum(loss_value)

        self.vae_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.actor.encoder.parameters(), 1, error_if_nonfinite=False)
        torch.nn.utils.clip_grad_norm_(self.policy.actor.parameters(), 1, error_if_nonfinite= False)

        self.vae_optimizer.step()
        self.beta_scheduler.step()

        res = {loss_name[i]: loss_value[i] for i in range(loss_num)}
        res['beta'] = self.beta_scheduler.value
        res['loss'] = loss
        return res
    
    
    def train_world_model(self, states, actions ):
        if not self._ensure_statics():
            return
        
        rollout_length = states.shape[1] -1
        loss_name = ['pos', 'rot', 'vel', 'avel']
        loss_num = len(loss_name)
        loss = list( ([] for _ in range(loss_num)) )
        states = states.transpose(0,1).contiguous().to(default_device())
        #state_muscle_lengths = state_muscle_lengths.transpose(0,1).contiguous().to(default_device())
        actions = actions.transpose(0,1).contiguous().to(default_device())
        cur_state = states[0]
        #cur_muscle_state = state_muscle_lengths[0]
        for i in range(rollout_length):
            next_state = states[i+1]
            #next_muscle_length = state_muscle_lengths[i+1]
            pred_next_state = self.policy.actor.world_model(cur_state, actions[i])
            #pred_muscle_length = get_muscle_lengths(pred_next_state.detach(), self.muscle_attachment_points, self.body_indices, self.muscle_indices) / self.muscle_length_scale
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
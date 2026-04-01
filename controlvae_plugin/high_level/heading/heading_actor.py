import json
from typing import Callable, List, Dict, Tuple, Optional, Union, Any
import abc

from mlagents.torch_utils import torch, nn, default_device
from mlagents.trainers.torch_entities.model_serialization import TensorNames
from mlagents_envs.base_env import ActionSpec, ObservationSpec, ObservationType
from mlagents.trainers.torch_entities.action_model import ActionModel
from mlagents.trainers.torch_entities.agent_action import AgentAction
from mlagents.trainers.settings import NetworkSettings, EncoderType, ConditioningType
from mlagents.trainers.torch_entities.utils import ModelUtils
from mlagents.trainers.torch_entities.decoders import ValueHeads
from mlagents.trainers.torch_entities.layers import LSTM, LinearEncoder
from mlagents.trainers.torch_entities.encoders import VectorInput
from mlagents.trainers.buffer import AgentBuffer
from mlagents.trainers.trajectory import ObsUtil
from mlagents.trainers.torch_entities.conditioning import ConditionalEncoder
from mlagents.trainers.torch_entities.attention import (
    EntityEmbedding,
    ResidualSelfAttention,
    get_zero_entities_mask,
)
from mlagents.trainers.exception import UnityTrainerException
from mlagents.trainers.torch_entities.networks import Actor

import random
from typing import List,Dict
from numpy import dtype
from torch import nn
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

from PlayGround.playground_util import *
from ...debug.debug import dump_actor

from ...settings import *
from ...actor import ControlVAEActor



class ControlVAEHeadingActor(ControlVAEActor):
    MODEL_EXPORT_VERSION = 3

    def __init__(
        self,
        seed: int,
        observation_spec: List[ObservationSpec],
        network_settings: NetworkSettings,
        action_spec: ActionSpec,
        statistics : torch.Tensor,
        dt : float = 0.02,
        **kargs
    ):
        super().__init__(seed, observation_spec, network_settings, action_spec, statistics, dt, **kargs)
        self.action_spec = action_spec
        self.version_number = torch.nn.Parameter(
            torch.Tensor([self.MODEL_EXPORT_VERSION]), requires_grad=False
        )
        self.continuous_act_size = int(self.action_spec.continuous_size)
        self.continuous_act_size_vector = torch.nn.Parameter(
            torch.Tensor([int(self.action_spec.continuous_size)]), requires_grad=False
        )
        self.discrete_act_size_vector = torch.nn.Parameter(
            torch.Tensor([self.action_spec.discrete_branches]), requires_grad=False
        )
        self.memory_size = 0
        self.memory_size_vector = torch.nn.Parameter(
            torch.Tensor([int(self.memory_size)]), requires_grad=False
        )
    
        self.observations_size_vector = observation_spec[0].shape[0]
        #obs is state sz + normalized state obs sz + target sz + normalized target obs sz \
        # + 1 done + 1 prior vs post bool \
        # + 2 for speed + angle
        self.done_sz = 1
        self.priorvpost_sz = 1
        self.heading_sz = 2
        extra = self.done_sz + self.priorvpost_sz + self.heading_sz

        self.state_obs_sz = (self.observations_size_vector - extra) // 2
        self.body_sz = (self.state_obs_sz - 3) // 29  #13 body + 16 body + 3
        self.state_sz = self.body_sz * 13
        self.obs_sz = self.body_sz * 16 + 3

        self.target_sz = self.state_sz
        self.target_obs_sz = self.obs_sz

        #not heading sz because it's used after trig conversion
        self.processed_heading_sz = 3
        self.task_ob_size = self.obs_sz + self.processed_heading_sz 

        self.high_level = ptu.build_mlp(self.task_ob_size, kargs["latent_size"], 3, 256, 'ELU').to(default_device())
        self.dance = False
        #modes
        self.mode = kargs["mode"]
        self.motion_type = kargs["type"]

        #trajectory debugging
        self.count = 0
        self._debug_traj_path = "debug_traj.pt"
        self._debug_traj = []


    def unpack_raw(self, raw, state_sz, obs_sz) -> dict:
        """
        raw: [B, obs_dim] coming from Unity (vector obs)
        packed as [state | target | done]
        """
        if isinstance(raw, torch.Tensor):
            if raw.dim() == 1:
                raw = raw.unsqueeze(0)
        elif isinstance(raw, np.ndarray):
            if raw.ndim == 1:
                raw = raw[None, :]

        N = state_sz
        M = obs_sz

        state = raw[:, 0:N] #
        state_obs = raw[:, N:N + M]
        target = raw[:, N + M:N + M + N]
        target_obs = raw[:, N + M + N:N + M + N + M]
        done = raw[:, N + M + N + M:N + M + N + M + 1]
        prior_vs_post = raw[:, N + M + N + M + 1:N + M + N + M + 2]
        heading = raw[:,N + M + N + M + 2:N + M + N + M + 2 + self.heading_sz]

        return {"state": state, "state_obs": state_obs, "target": target, "target_obs": target_obs, \
                 "done": done, "prior_vs_post": prior_vs_post, "heading": heading}


    @staticmethod
    def target2n_target(state, target):
        if len(state.shape) ==2:
            state = state[None,...]
        if len(target.shape) ==1:
            target = target[None,...]
        if isinstance(target, np.ndarray):
            target = ptu.from_numpy(target)
        if isinstance(state, np.ndarray):
            state = ptu.from_numpy(state)
        facing_direction = get_root_facing(state)
        facing_angle = torch.arctan2(facing_direction[:,2], facing_direction[:,0])
        delta_angle = target[:,1] - facing_angle
        res = torch.cat([target[:,0, None], torch.cos(delta_angle[:,None]), torch.sin(delta_angle[:,None])], dim = -1)
        return res

    def get_action_and_stats(
        self,
        inputs: List[torch.Tensor],
        masks: Optional[torch.Tensor] = None,
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
    ) -> Tuple[AgentAction, Dict[str, Any], torch.Tensor]:

        #convert
        obs = inputs[0].to(default_device())
        packed = self.unpack_raw(obs, self.state_sz, self.obs_sz)
        state = packed["state"]
        state_obs = packed["state_obs"]
        target = packed["target"]
        target_obs = packed["target_obs"]
        done = packed["done"]
        prior_vs_post = packed["prior_vs_post"]
        heading = packed["heading"]

        batch = obs.shape[0]
        state = state.reshape(batch, self.body_sz, 13)
        target = target.reshape(batch, self.body_sz, 13)

        """
        if self.count <= 512:
            self._debug_traj.append([x.detach().cpu() for x in inputs])
        if self.count == 512:
            torch.save(self._debug_traj, self._debug_traj_path)
            a = 0
        """
        assert(self.motion_type == ControlVAEType.HEADING)
        
        if self.motion_type == ControlVAEType.HEADING:
            if self.mode == ControlVAEMode.TRAINING:
                latent, mu, _ = self.encoder.encode_prior(state_obs)
                heading_target = self.target2n_target(state, heading)
                task = torch.cat([state_obs, heading_target], dim=1)
                offset = self.high_level(task)
                latent = mu+offset
                action = self.decode(state_obs, latent)

            elif self.mode == ControlVAEMode.INFERENCE:
                latent, mu, _ = self.encoder.encode_prior(state_obs)
                heading_target = self.target2n_target(state, heading)
                task = torch.cat([state_obs, heading_target], dim=1)
                offset = self.high_level(task)
                latent = mu+offset
                action = self.decode(state_obs, latent)

        # convert to AgentAction and run_out
        act = AgentAction(continuous_tensor=action.detach().clone(), discrete_list=None)
        run_out = {"env_action": act.to_action_tuple(clip=True),
                    #"mu_prior": mu_prior.detach().clone(),
                    #"mu_post": mu_post.detach().clone(),
                   }
        
        self.count += 1

        return act, run_out, memories 

    def heading_from_state(self, state, target):
        """
        state: [B, state_sz] (flat)
        target: [B, 2] -> [speed, angle]

        uses first body (first 13 values)

        returns: [B, 3] -> [speed, cos(delta), sin(delta)]
        """

        # --- root rotation (first body, indices 3:7) ---
        quat = state[:, 3:7]  # [B, 4]

        # forward vector (Z axis)
        forward = torch.zeros_like(quat[:, :3])
        forward[:, 2] = 1.0

        # quaternion rotate forward vector
        q_xyz = quat[:, :3]
        q_w = quat[:, 3:4]

        t = 2.0 * torch.cross(q_xyz, forward, dim=1)
        facing = forward + q_w * t + torch.cross(q_xyz, t, dim=1)

        # project to XZ plane
        facing_xz = facing[:, [0, 2]]

        # normalize (avoid NaNs)
        facing_xz = facing_xz / (torch.norm(facing_xz, dim=-1, keepdim=True) + 1e-8)

        # facing angle
        facing_angle = torch.atan2(facing_xz[:, 1], facing_xz[:, 0])

        # --- target ---
        target_speed = target[:, 0]
        target_angle = target[:, 1]

        delta = target_angle - facing_angle

        return torch.cat([
            target_speed[:, None],
            torch.cos(delta)[:, None],
            torch.sin(delta)[:, None]
        ], dim=-1)

    def forward(
        self,
        inputs: List[torch.Tensor],
        masks: Optional[torch.Tensor] = None,
        memories: Optional[torch.Tensor] = None,
    ) -> Tuple[Union[int, torch.Tensor], ...]:
        """
        Forward pass of the Actor for inference. This is required for export to ONNX, and
        the inputs and outputs of this method should not be changed without a respective change
        in the ONNX export code.
        """
        obs = inputs[0].to(default_device())
        packed = self.unpack_raw(obs, self.state_sz, self.obs_sz)
        state = packed["state"]
        state_obs = packed["state_obs"]
        target = packed["target"]
        target_obs = packed["target_obs"]
        done = packed["done"]
        prior_vs_post = packed["prior_vs_post"]
        heading = packed["heading"]

        #print("[ACTOR] before encode prior")
        latent, mu, _ = self.encoder.encode_prior(state_obs, deterministic=True)
        #print("[ACTOR] before heading from state")
        heading_target = self.heading_from_state(state, heading)
        #print("[ACTOR] after heading from state")
        
        task = torch.cat([state_obs, heading_target], dim=1)
        #print("[ACTOR] before offset")
        offset = self.high_level(task)
        #latent = mu+offset
        #print("[ACTOR] before decode")
        action = self.decode(state_obs, latent)
        #print("[ACTOR] after decode")

        #Needs to be in [obs, act size]
        batch = obs.shape[0]
        action = action.reshape(batch, self.continuous_act_size)

        #for some reason the version number, memory size and shape from above return 0
        version_tensor = torch.tensor([self.MODEL_EXPORT_VERSION], dtype=torch.float32, device=obs.device)
        memory_tensor = torch.tensor([self.memory_size], dtype=torch.float32, device=obs.device)
        shape_tensor = torch.tensor([self.continuous_act_size], dtype=torch.float32, device=obs.device)

        #exporting
        export_out = [version_tensor, memory_tensor]

        #don't let the names fool you, it's actually prior deterministic in action, and posterior deterministic in action deterministic
        if self.action_spec.continuous_size > 0:
            export_out += [
                action,
                shape_tensor,
                action,
            ]

        #memories drama
        device = obs.device
        if self.memory_size > 0:
            mem_out = memories
            if mem_out is None:
                mem_out = torch.zeros((1, self.memory_size), device=device, dtype=obs.dtype)
            export_out.append(mem_out)

        #print("[ACTOR] forward before return")
        return tuple(export_out)
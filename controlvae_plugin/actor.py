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

from .debug.debug import dump_actor

from .settings import *

class ControlVAEActor(nn.Module, Actor):
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
        super().__init__()
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
        #obs is state sz + normalized state obs sz + target sz + normalized target obs sz + 1 done + 1 prior vs post bool
        self.state_obs_sz = (self.observations_size_vector - 2) // 2
        self.body_sz = (self.state_obs_sz - 3) // 29  #13 body + 16 body + 3
        self.state_sz = self.body_sz * 13
        self.obs_sz = self.body_sz * 16 + 3

        self.target_sz = self.state_sz
        self.target_obs_sz = self.obs_sz

        self.action_sigma = kargs["action_sigma"]
        self.with_noise = kargs["with_noise"]

        self.encoder = ONXXSimpleLearnablePriorEncoder(
            input_size= self.obs_sz,
            condition_size= self.obs_sz,
            output_size= kargs['latent_size'],
            fix_var = kargs['encoder_fix_var'],
            **kargs
        ).to(default_device())
        
        self.agent = ONNXGatingMixedDecoder(
            # latent_size= kargs['latent_size'],
            condition_size= self.obs_sz,
            output_size=self.continuous_act_size,
            **kargs
        ).to(default_device())
        
        # 6 per body
        self.delta_sz = self.body_sz * 6
        self.dt = dt

        self.world_model = SimpleWorldModel(self.obs_sz, self.continuous_act_size, self.delta_sz, self.dt, statistics,  **kargs).to(default_device())
        
        #modes
        self.mode = kargs["mode"]
        self.motion_type = kargs["type"]

        #trajectory debugging
        self.count = 0
        self._debug_traj_path = "debug_traj.pt"
        self._debug_traj = []

    def encode(self, normalized_obs, normalized_target, deterministic = False):
        """encode observation and target into posterior distribution

        Args:
            normalized_obs (Optional[Tensor,np.ndarray]): normalized current observation
            normalized_target (Optional[Tensor, np.ndarray]): normalized current target 

        Returns:
            Tuple(tensor, tensor, tensor): 
                latent coder, mean of prior distribution, mean of posterior distribution 
        """
        return self.encoder(normalized_obs, normalized_target, deterministic = deterministic)
    
    def decode(self, normalized_obs, latent):
        """decode latent code into action space

        Args:
            normalized_obs (tensor): normalized current observation
            latent (tensor): latent code

        Returns:
            tensor: action
        """
        action = self.agent(latent, normalized_obs)        
        return action
    
    def normalize_obs(self, observation):
        if isinstance(observation, np.ndarray):
            observation = ptu.from_numpy(observation)
        if len(observation.shape) == 1:
            observation = observation[None,...]
        return ptu.normalize(observation, self.obs_mean, self.obs_std)
    
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

        return {"state": state, "state_obs": state_obs, "target": target, "target_obs": target_obs, "done": done, "prior_vs_post": prior_vs_post}

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
    
    def act_tracking(self, **obs_info):
        """
        try to track reference motion
        """
        target = obs_info['target']
        
        n_target = self.normalize_obs(target)
        n_observation = self.obsinfo2n_obs(obs_info)
        
        latent_code, mu_post, mu_prior = self.encode(n_observation, n_target)
        action = self.decode(n_observation, latent_code)
        info = {
            "mu_prior": mu_prior,
            "mu_post": mu_post
        }
        return action, info
    
    def act_prior(self, obs_info):
        """
        try to track reference motion
        """
        n_observation = self.obsinfo2n_obs(obs_info)
        latent_code, mu_prior, logvar = self.encoder.encode_prior(n_observation)
        action = self.decode(n_observation, latent_code)
        
        return action
    
    def act_determinastic(self, obs_info):
        action, _ = self.act_tracking(**obs_info)
        return action
                
    def act_distribution(self, obs_info):
        """
        Add noise to the output action
        """
        action = self.act_determinastic(obs_info)
        action_distribution = D.Independent(D.Normal(action, self.action_sigma), -1)
        return action_distribution
    
    def update_normalization(self, buffer: AgentBuffer) -> None:
        """
        Updates normalization of Actor based on the provided List of vector obs.
        :param vector_obs: A List of vector obs as tensors.
        """
        pass

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

        
        if self.count <= 512:
            self._debug_traj.append([x.detach().cpu() for x in inputs])
        if self.count == 512:
            torch.save(self._debug_traj, self._debug_traj_path)
            a = 0

        if self.mode == ControlVAEMode.TRAINING:
            latent_code, mu_post, mu_prior = self.encode(state_obs, target_obs, deterministic=False)
            action_tracking = self.decode(state_obs, latent_code)

            if self.with_noise == True:
                action_tracking = D.Independent(D.Normal(action_tracking, self.action_sigma), -1)
                action_tracking = action_tracking.sample()

            #random action from distribution for stability
            if np.random.choice([True, False], p = [0.4, 0.6]):
                latent_code, mu, logvar = self.encoder.encode_prior(state_obs, deterministic = False)
                #not actual tracking, it's prior sampling for exploration
                action_tracking = self.decode(state_obs, latent_code)                
                action_tracking = action_tracking + torch.randn_like(action_tracking) * 0.05

        """
        inference with python
        """
        if self.mode == ControlVAEMode.INFERENCE:
            if self.motion_type == ControlVAEType.TRACKING:
                latent_code, mu_post, mu_prior = self.encode(state_obs, target_obs, deterministic=False)
                action_tracking = self.decode(state_obs, latent_code)
            elif self.motion_type == ControlVAEType.PRIOR:
                #actually prior
                latent_code, mu, logvar = self.encoder.encode_prior(state_obs, deterministic = False)
                action_tracking = self.decode(state_obs, latent_code) 

        # convert to AgentAction and run_out
        act = AgentAction(continuous_tensor=action_tracking.detach().clone(), discrete_list=None)
        run_out = {"env_action": act.to_action_tuple(clip=True),
                    #"mu_prior": mu_prior.detach().clone(),
                    #"mu_post": mu_post.detach().clone(),
                   }
        
        self.count += 1

        return act, run_out, memories 

    def get_stats(
        self,
        inputs: List[torch.Tensor],
        actions: AgentAction,
        masks: Optional[torch.Tensor] = None,
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
    ) -> Dict[str, Any]:
        """
        Returns log_probs for actions and entropies.
        If memory is enabled, return the memories as well.
        :param inputs: A List of inputs as tensors.
        :param actions: AgentAction of actions.
        :param masks: If using discrete actions, a Tensor of action masks.
        :param memories: If using memory, a Tensor of initial memories.
        :param sequence_length: If using memory, the sequence length.
        :return: A Tuple of AgentAction, ActionLogProbs, entropies, and memories.
            Memories will be None if not using memory.
        """

        pass

    #@abc.abstractmethod
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

        
        #tracking
        latent_code, mu_post, mu_prior = self.encode(state_obs, target_obs)
        action_tracking = self.decode(state_obs, latent_code)
        
        #random action from distribution
        latent, mu, logvar = self.encoder.encode_prior(state_obs)
        action_prior = self.decode(state_obs, latent)


        #deterministic prior
        latent_deterministic, _, _ = self.encoder.encode_prior(state_obs, deterministic=True)
        prior_deterministic = self.decode(state_obs, latent_deterministic)

        #determinism
        latent_post_deterministic, _, _ = self.encoder.encode_post(state_obs, target_obs, deterministic=True)
        posterior_deterministic = self.decode(state_obs, latent_post_deterministic + latent_deterministic)

        

        #print("[ACTOR FORWARD] post vs deterministic: ",(action_tracking - posterior_deterministic).abs().max())

        #if equivalent
        flag = prior_vs_post > 0.5
        #action = torch.where(flag, action_tracking, action_prior).float()
        #action_deterministic = torch.where(flag, posterior_deterministic, prior_deterministic).float()

        batch = obs.shape[0]

        #Needs to be in [obs, act size]
        action = prior_deterministic.reshape(batch, self.continuous_act_size)
        action_deterministic = posterior_deterministic.reshape(batch, self.continuous_act_size)

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
                action_deterministic,
            ]

        #memories drama
        device = obs.device
        if self.memory_size > 0:
            mem_out = memories
            if mem_out is None:
                mem_out = torch.zeros((1, self.memory_size), device=device, dtype=obs.dtype)
            export_out.append(mem_out)


        return tuple(export_out)
    



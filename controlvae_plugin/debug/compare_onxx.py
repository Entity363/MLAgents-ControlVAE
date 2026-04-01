import torch
import numpy as np
import onnxruntime as ort

import random
from typing import Any, List,Dict
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
import pickle

# -------------------------
# PATHS
# -------------------------
DUMP_PATH = f"D:/Documents/Unity/ml-agents-plugins/ControlVAE-Plugin/debug_traj.pt"
ONNX_PATH = f"D:/Documents/Unity/ml-agents-plugins/ControlVAE-Plugin/results/ppo/ControlVAE.onnx"
DATA_PATH = f"D:/Documents/Unity/ml-agents-plugins/ControlVAE-Plugin/results/ppo/checkpoint.data"

# -------------------------
# LOAD TRAJECTORY
# -------------------------
traj = torch.load(DUMP_PATH, map_location="cpu")
inputs = traj[0]  # pick a step
obs = inputs[0].float()

print("obs shape:", obs.shape)

# -------------------------
# SPLIT (same as actor)
# -------------------------
def unpack_raw(raw, state_sz, obs_sz):
    N = state_sz
    M = obs_sz

    state = raw[:, 0:N]
    state_obs = raw[:, N:N + M]
    target = raw[:, N + M:N + M + N]
    target_obs = raw[:, N + M + N:N + M + N + M]
    done = raw[:, N + M + N + M:N + M + N + M + 1]
    prior_vs_post = raw[:, N + M + N + M + 1:N + M + N + M + 2]

    return {
        "state": state,
        "state_obs": state_obs,
        "target": target,
        "target_obs": target_obs,
        "done": done,
        "prior_vs_post": prior_vs_post,
    }

# -------------------------
# CONSTANTS
# -------------------------
state_sz = 260
obs_sz = 323
continuous_act_size = 76
MODEL_EXPORT_VERSION = 3
memory_size = 0

packed = unpack_raw(obs, state_sz, obs_sz)

state = packed["state"]
state_obs = packed["state_obs"]
target = packed["target"]
target_obs = packed["target_obs"]
done = packed["done"]
prior_vs_post = packed["prior_vs_post"]

kargs: Dict[str, Any] = {
    "action_sigma": 0.05,
    "with_noise": True,
    "latent_size": 64,
    "encoder_fix_var": 0.3,
    "encoder_hidden_layer_num": 2,
    "encoder_hidden_layer_size": 512,
    "encoder_activation": "ELU",
    "actor_hidden_layer_size": 512,
    "actor_hidden_layer_num": 3,
    "actor_activation": "ELU",
    "actor_num_experts": 6,
    "actor_gate_hidden_layer_size": 64,
}

# -------------------------
# BUILD MODULES
# -------------------------
encoder = ONXXSimpleLearnablePriorEncoder(
    input_size=obs_sz,
    condition_size=obs_sz,
    output_size=kargs["latent_size"],
    fix_var=kargs["encoder_fix_var"],
    **kargs
).to(default_device())

decoder = ONXXGatingMixedDecoder(
    condition_size=obs_sz,
    output_size=continuous_act_size,
    **kargs
).to(default_device())

# -------------------------
# LOAD WEIGHTS
# -------------------------
ckpt = torch.load(DATA_PATH, map_location="cpu")
policy_state = ckpt["Policy"]

encoder_state = {}
for k, v in policy_state.items():
    if k.startswith("encoder."):
        encoder_state[k.replace("encoder.", "")] = v.to(default_device())
encoder.load_state_dict(encoder_state, strict=True)

decoder_state = {}
for k, v in policy_state.items():
    if k.startswith("agent."):
        decoder_state[k.replace("agent.", "")] = v.to(default_device())
decoder.load_state_dict(decoder_state, strict=True)

encoder.eval()
decoder.eval()

device = next(encoder.parameters()).device

obs = obs.to(device)
state_obs = state_obs.to(device)
target_obs = target_obs.to(device)
prior_vs_post = prior_vs_post.to(device)

# -------------------------
# PYTORCH FORWARD (EXACT ACTOR LOGIC)
# -------------------------
with torch.no_grad():
    # tracking
    latent_code, mu_post, mu_prior = encoder(state_obs, target_obs)
    action_tracking = decoder(latent_code, state_obs)

    # random action from distribution
    latent, mu, logvar = encoder.encode_prior(state_obs)
    action_prior = decoder(latent, state_obs)

    # determinism
    latent_deterministic, _, _ = encoder.encode_prior(state_obs, deterministic=True)
    prior_deterministic = decoder(latent_deterministic, state_obs)

    latent_post_deterministic, _, _ = encoder.encode_post(
        state_obs, target_obs, deterministic=True
    )
    posterior_deterministic = decoder(
        latent_post_deterministic + latent_deterministic, state_obs
    )

    print(
        "[PYTORCH] post vs deterministic:",
        (action_tracking - posterior_deterministic).abs().max().item()
    )

    # if equivalent
    """flag = prior_vs_post > 0.5
    action = torch.where(flag, action_tracking, action_prior).float()
    action_deterministic = torch.where(
        flag, posterior_deterministic, prior_deterministic
    ).float()"""

    batch = obs.shape[0]
    action = posterior_deterministic.reshape(batch, continuous_act_size)
    action_deterministic = posterior_deterministic.reshape(batch, continuous_act_size)

pt_action = action.detach().cpu().numpy()
pt_action_det = action_deterministic.detach().cpu().numpy()

# -------------------------
# ONNX
# -------------------------
sess = ort.InferenceSession(ONNX_PATH, providers=["GPUExecutionProvider"])

print("ONNX inputs:", [i.name for i in sess.get_inputs()])
print("ONNX outputs:", [o.name for o in sess.get_outputs()])

feed = {
    "obs_0": obs.detach().cpu().numpy()
}

onnx_out = sess.run(None, feed)
onnx_action = onnx_out[2]
onnx_action_det = onnx_out[4]

# -------------------------
# COMPARE
# -------------------------
def report(name, a, b):
    diff = np.abs(a - b)
    print(f"\n=== {name} ===")
    print("max abs diff :", diff.max())
    print("mean abs diff:", diff.mean())

report("STOCHASTIC ACTION", onnx_action, pt_action)
report("DETERMINISTIC ACTION", onnx_action_det, pt_action_det)

print("\nPT det first 8 :", pt_action_det[0, :8])
print("ONNX det first 8:", onnx_action_det[0, :8])
print("PT act first 8 :", pt_action[0, :8])
print("ONNX act first 8:", onnx_action[0, :8])
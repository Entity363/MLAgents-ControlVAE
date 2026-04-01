from typing import Any, Dict, List, Optional
import numpy as np
from mlagents.torch_utils import torch, default_device
import copy

from mlagents.trainers.action_info import ActionInfo
from mlagents.trainers.behavior_id_utils import get_global_agent_id
from mlagents.trainers.policy import Policy
from mlagents_envs.base_env import DecisionSteps, BehaviorSpec
from mlagents_envs.timers import timed

from mlagents.trainers.settings import NetworkSettings
from mlagents.trainers.torch_entities.networks import GlobalSteps

from mlagents.trainers.torch_entities.utils import ModelUtils

from abc import abstractmethod
from typing import Dict, List, Optional
import numpy as np

from mlagents_envs.base_env import ActionTuple, BehaviorSpec, DecisionSteps
from mlagents_envs.exception import UnityException

from mlagents.trainers.action_info import ActionInfo
from mlagents.trainers.settings import NetworkSettings
from mlagents.trainers.behavior_id_utils import GlobalAgentId

from .debug.debug import dump_actor, watch_param

#apparently same as torch policy. Might want to add prior and posterior to stats
class ControlVAEPolicy(Policy):
    def __init__(
        self,
        seed: int,
        behavior_spec: BehaviorSpec,
        network_settings: NetworkSettings,
        actor : type,
        statistics : Dict[str, torch.Tensor],
        actor_kwargs: Dict[str, Any]):
            super().__init__(seed, behavior_spec, network_settings)
            #self.global_step = (GlobalSteps())
            self.iteration = 0
            self.stats_name_to_update_name = {
                "Losses/Value Loss": "value_loss",
                "Losses/Policy Loss": "policy_loss",
            }

            self.statistics = statistics

            self.actor = actor(
                seed = seed,
                observation_spec=self.behavior_spec.observation_specs,
                network_settings=network_settings,
                action_spec=behavior_spec.action_spec,
                dt = 0.02,
                statistics = self.statistics,
                **actor_kwargs,
            )

            
            #watch_param(self.actor.encoder.post.mu.weight, "post.mu.weight")
            #watch_param(self.actor.encoder.prior.mu.weight, "prior.mu.weight")
    

            # Save the m_size needed for export
            self._export_m_size = self.m_size
            # m_size needed for training is determined by network, not trainer settings
            self.m_size = self.actor.memory_size

            self.actor.to(default_device())

            self.count = 0

    @property
    def export_memory_size(self) -> int:
        """
        Returns the memory size of the exported ONNX policy. This only includes the memory
        of the Actor and not any auxillary networks.
        """
        return self._export_m_size

    def _extract_masks(self, decision_requests: DecisionSteps) -> np.ndarray:
        device = default_device()
        mask = None
        if self.behavior_spec.action_spec.discrete_size > 0:
            num_discrete_flat = np.sum(self.behavior_spec.action_spec.discrete_branches)
            mask = torch.ones(
                [len(decision_requests), num_discrete_flat], device=device
            )
            if decision_requests.action_mask is not None:
                mask = torch.as_tensor(
                    1 - np.concatenate(decision_requests.action_mask, axis=1),
                    device=device,
                )
        return mask
    
    @timed
    def evaluate(
        self, decision_requests: DecisionSteps, global_agent_ids: List[str]
    ) -> Dict[str, Any]:
        """
        Evaluates policy for the agent experiences provided.
        :param global_agent_ids:
        :param decision_requests: DecisionStep object containing inputs.
        :return: Outputs from network as defined by self.inference_dict.
        """
        obs = decision_requests.obs
        masks = self._extract_masks(decision_requests)
        device = default_device()
        tensor_obs = [torch.as_tensor(np_ob, device=device) for np_ob in obs]

        memories = torch.as_tensor(
            self.retrieve_memories(global_agent_ids), device=device
        ).unsqueeze(0)
        with torch.no_grad():
            action, run_out, memories = self.actor.get_action_and_stats(
                tensor_obs, masks=masks, memories=memories
            )
        run_out["action"] = action.to_action_tuple()
        if "log_probs" in run_out:
            run_out["log_probs"] = run_out["log_probs"].to_log_probs_tuple()
        if "entropy" in run_out:
            run_out["entropy"] = ModelUtils.to_numpy(run_out["entropy"])
        if self.use_recurrent:
            run_out["memory_out"] = ModelUtils.to_numpy(memories).squeeze(0)
        return run_out

    def get_action(
        self, decision_requests: DecisionSteps, worker_id: int = 0
    ) -> ActionInfo:
        """
        Decides actions given observations information, and takes them in environment.
        :param worker_id:
        :param decision_requests: A dictionary of behavior names and DecisionSteps from environment.
        :return: an ActionInfo containing action, memories, values and an object
        to be passed to add experiences
        """

        if len(decision_requests) == 0:
            return ActionInfo.empty()

        global_agent_ids = [
            get_global_agent_id(worker_id, int(agent_id))
            for agent_id in decision_requests.agent_id
        ]  # For 1-D array, the iterator order is correct.

        run_out = self.evaluate(decision_requests, global_agent_ids)

        self.save_memories(global_agent_ids, run_out.get("memory_out"))

        self.check_nan_action(run_out.get("action"))
       
        action = ActionInfo(
            action=run_out.get("action"),
            env_action=run_out.get("env_action"),
            outputs=run_out,
            agent_ids=list(decision_requests.agent_id),
        )
        return action

    def get_current_step(self):
        """
        Gets current model step.
        :return: current model step.
        """
        return self.iteration

    def set_step(self, step: int) -> int:
        """
        Sets current model step to step without creating additional ops.
        :param step: Step to set the current model step to.
        :return: The step the model was set to.
        """
        self.step = step 
        return step

    def increment_step(self, n_steps):
        """
        Increments model step.
        """
        self.iteration += 1
        return self.get_current_step()

    def get_weights(self):
        return {k: v.detach().cpu().clone() for k, v in self.actor.state_dict().items()}

    def load_weights(self, values):
        self.actor.load_state_dict(values, strict=False)

    def get_modules(self):
        return {"Policy": self.actor} #, "global_step": self.global_step

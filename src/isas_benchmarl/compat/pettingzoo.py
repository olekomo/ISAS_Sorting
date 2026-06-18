"""Optional PettingZoo compatibility adapter (not used by BenchMARL training)."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
import torch
from gymnasium.spaces import Box, Discrete
from pettingzoo import ParallelEnv
from tensordict import TensorDict

from ..env import ISASTorchEnv


class ISASSortingEnv(ParallelEnv):
    """B=1 compatibility surface matching the previous PettingZoo class name."""

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "isas_sorting_v1",
        "is_parallelizable": True,
    }

    def __init__(self, config_dict: Any, action_type: str = "discrete"):
        super().__init__()
        if action_type != "discrete":
            raise ValueError("The compatibility adapter supports discrete actions only.")
        self.torch_env = ISASTorchEnv(config_dict, num_envs=1, device="cpu")
        self.config = self.torch_env.config
        self.possible_agents = self.torch_env.group_map["actors"]
        self.agents = self.possible_agents.copy()
        self._observation_spaces = {
            agent: Box(
                -np.inf,
                np.inf,
                shape=(self.torch_env.observation_dim,),
                dtype=np.float32,
            )
            for agent in self.possible_agents
        }
        self._action_spaces = {
            agent: Discrete(self.config.num_actions) for agent in self.possible_agents
        }
        self.state_space = Box(
            -np.inf,
            np.inf,
            shape=(self.torch_env.state_dim,),
            dtype=np.float32,
        )
        self._current = None

    @lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self._observation_spaces[agent]

    @lru_cache(maxsize=None)
    def action_space(self, agent):
        return self._action_spaces[agent]

    def _decode(self, td):
        obs = td["actors", "observation"][0].detach().cpu().numpy()
        mask = td["actors", "action_mask"][0].detach().cpu().numpy()
        observations = {
            agent: obs[index].copy()
            for index, agent in enumerate(self.possible_agents)
            if agent in self.agents
        }
        infos = {
            agent: {"action_mask": mask[index].copy()}
            for index, agent in enumerate(self.possible_agents)
            if agent in self.agents
        }
        return observations, infos

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.torch_env.set_seed(int(seed))
        self.agents = self.possible_agents.copy()
        self._current = self.torch_env.reset()
        return self._decode(self._current)

    def step(self, actions):
        if not self.agents:
            return {}, {}, {}, {}, {}
        action_tensor = torch.zeros(
            (1, len(self.possible_agents)), dtype=torch.int64
        )
        for index, agent in enumerate(self.possible_agents):
            if agent in actions:
                action_tensor[0, index] = int(np.asarray(actions[agent]).item())
        input_td = self._current.clone(False)
        input_td.set(
            ("actors", "action"),
            action_tensor,
        )
        transition = self.torch_env.step(input_td)
        self._current = transition["next"]
        observations, infos = self._decode(self._current)
        reward = self._current["actors", "reward"][0, :, 0].detach().cpu().numpy()
        terminated = bool(self._current["terminated"][0, 0].item())
        truncated = bool(self._current["truncated"][0, 0].item())
        active_agents = self.agents.copy()
        rewards = {agent: float(reward[index]) for index, agent in enumerate(active_agents)}
        terminations = {agent: terminated for agent in active_agents}
        truncations = {agent: truncated for agent in active_agents}
        if terminated or truncated:
            self.agents = []
        return observations, rewards, terminations, truncations, infos

    def state(self):
        if self._current is None:
            raise RuntimeError("Call reset() before state().")
        return self._current["state"][0].detach().cpu().numpy().copy()

    def render(self):
        return self.torch_env.render()

    def close(self):
        self.torch_env.close()


def parallel_env(config_dict: Any, action_type: str = "discrete", **kwargs):
    return ISASSortingEnv(config_dict=config_dict, action_type=action_type)


env = parallel_env

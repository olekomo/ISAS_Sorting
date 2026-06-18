"""BenchMARL task adapter for the native TorchRL environment."""

from __future__ import annotations

import copy
from typing import Callable, Dict, List, Optional

import torch
from tensordict import TensorDictBase
from torchrl.data import Composite
from torchrl.envs import EnvBase

from benchmarl.environments.common import Task, TaskClass
from benchmarl.utils import DEVICE_TYPING

from ..env import ISASTorchEnv
from ..rewards import METRIC_NAMES


class ISASClass(TaskClass):
    """Construct and describe ISAS environments for any BenchMARL algorithm."""

    def get_env_fun(
        self,
        num_envs: int,
        continuous_actions: bool,
        seed: Optional[int],
        device: DEVICE_TYPING,
    ) -> Callable[[], EnvBase]:
        if continuous_actions:
            raise ValueError(
                "The current ISAS interface is delayed discrete. Set "
                "prefer_continuous_actions=False in the BenchMARL experiment."
            )
        config = copy.deepcopy(self.config)
        return lambda: ISASTorchEnv(
            config=config,
            num_envs=num_envs,
            device=device,
            seed=seed,
        )

    def supports_continuous_actions(self) -> bool:
        return False

    def supports_discrete_actions(self) -> bool:
        return True

    def max_steps(self, env: EnvBase) -> int:
        return env.config.max_decision_steps

    def has_render(self, env: EnvBase) -> bool:
        return True

    def group_map(self, env: EnvBase) -> Dict[str, List[str]]:
        return env.group_map

    def observation_spec(self, env: EnvBase) -> Composite:
        spec = env.full_observation_spec_unbatched.clone()
        del spec[(env.group_name, "info")]
        del spec[(env.group_name, "action_mask")]
        del spec["state"]
        return spec

    def info_spec(self, env: EnvBase) -> Optional[Composite]:
        source = env.full_observation_spec_unbatched
        return Composite(
            {
                env.group_name: Composite(
                    info=source[(env.group_name, "info")].clone(),
                    shape=source[env.group_name].shape,
                    device=source.device,
                )
            },
            device=source.device,
        )

    def state_spec(self, env: EnvBase) -> Optional[Composite]:
        source = env.full_observation_spec_unbatched
        return Composite(state=source["state"].clone(), device=source.device)

    def action_spec(self, env: EnvBase) -> Composite:
        return env.full_action_spec_unbatched.clone()

    def action_mask_spec(self, env: EnvBase) -> Optional[Composite]:
        source = env.full_observation_spec_unbatched
        return Composite(
            {
                env.group_name: Composite(
                    action_mask=source[(env.group_name, "action_mask")].clone(),
                    shape=source[env.group_name].shape,
                    device=source.device,
                )
            },
            device=source.device,
        )

    @staticmethod
    def env_name() -> str:
        return "isas"

    @staticmethod
    def log_info(batch: TensorDictBase) -> Dict[str, float]:
        # BenchMARL logs fields below (group, info) automatically. Keeping this
        # hook empty avoids reporting the same cumulative metrics twice.
        return {}

    @staticmethod
    def render_callback(experiment, env: EnvBase, data: TensorDictBase) -> torch.Tensor:
        frame = env.render()
        return torch.from_numpy(frame)


class ISASTask(Task):
    SORTING = None

    @staticmethod
    def associated_class():
        return ISASClass


# Compatibility alias for the previous project import name.
ISASSortingClass = ISASClass

"""Native TorchRL environment for the vectorized ISAS simulator."""

from __future__ import annotations

from typing import Any

import torch
from tensordict import TensorDict, TensorDictBase
from torchrl.data import Categorical, Composite, Unbounded
from torchrl.envs import EnvBase

from .config import ISASConfig, load_config
from .dynamics import ISASDynamics
from .rewards import METRIC_NAMES
from .state import SimulationEvents, SimulationState


class ISASTorchEnv(EnvBase):
    """Batched multi-agent sorting environment.

    The leading environment batch is native: one Python object simulates
    ``num_envs`` independent sorting systems on the selected device.
    """

    batch_locked = True
    group_name = "actors"

    def __init__(
        self,
        config: ISASConfig | Any,
        num_envs: int = 1,
        device: torch.device | str = "cpu",
        seed: int | None = None,
        render_env_index: int = 0,
    ):
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        self.config = load_config(config)
        self.num_envs = int(num_envs)
        self.num_actors = self.config.num_actors
        self.render_env_index = int(render_env_index)
        if not 0 <= self.render_env_index < self.num_envs:
            raise ValueError("render_env_index is outside the environment batch")

        super().__init__(
            device=torch.device(device),
            batch_size=torch.Size([self.num_envs]),
            run_type_checks=False,
        )

        self.group_map = {
            self.group_name: [f"actor_{index}" for index in range(self.num_actors)]
        }
        self.dynamics = ISASDynamics(self.config, self.device)
        self._state: SimulationState = self.dynamics.empty_state(self.num_envs)
        self._prediction_generator = torch.Generator(device=self.device)
        self._initialization_generator = torch.Generator(device=self.device)
        self._last_events: SimulationEvents | None = None
        self._renderer = None
        self._make_specs()
        self.set_seed(self.config.seed if seed is None else seed)

    @property
    def observation_dim(self) -> int:
        return self.dynamics.observations.obs_dim

    @property
    def state_dim(self) -> int:
        return self.dynamics.observations.state_dim

    def _make_specs(self) -> None:
        B, A = self.num_envs, self.num_actors
        device = self.device
        actors_shape = torch.Size([B, A])
        batch_shape = torch.Size([B])

        actor_observation_spec = Composite(
            observation=Unbounded(
                shape=(*actors_shape, self.observation_dim),
                dtype=torch.float32,
                device=device,
            ),
            action_mask=Categorical(
                2,
                shape=(*actors_shape, self.config.num_actions),
                dtype=torch.bool,
                device=device,
            ),
            info=Composite(
                {
                    name: Unbounded(
                        shape=(*actors_shape, 1),
                        dtype=torch.float32,
                        device=device,
                    )
                    for name in METRIC_NAMES
                },
                shape=actors_shape,
                device=device,
            ),
            shape=actors_shape,
            device=device,
        )
        self.observation_spec = Composite(
            {
                self.group_name: actor_observation_spec,
                "state": Unbounded(
                    shape=(*batch_shape, self.state_dim),
                    dtype=torch.float32,
                    device=device,
                ),
            },
            shape=batch_shape,
            device=device,
        )

        self.action_spec = Composite(
            {
                self.group_name: Composite(
                    action=Categorical(
                        self.config.num_actions,
                        shape=actors_shape,
                        dtype=torch.int64,
                        device=device,
                    ),
                    shape=actors_shape,
                    device=device,
                )
            },
            shape=batch_shape,
            device=device,
        )

        self.reward_spec = Composite(
            {
                self.group_name: Composite(
                    reward=Unbounded(
                        shape=(*actors_shape, 1),
                        dtype=torch.float32,
                        device=device,
                    ),
                    shape=actors_shape,
                    device=device,
                )
            },
            shape=batch_shape,
            device=device,
        )

        done_leaf = Categorical(
            2, shape=(*batch_shape, 1), dtype=torch.bool, device=device
        )
        self.done_spec = Composite(
            done=done_leaf,
            terminated=done_leaf.clone(),
            truncated=done_leaf.clone(),
            shape=batch_shape,
            device=device,
        )

    def _set_seed(self, seed: int) -> int:
        self._prediction_generator.manual_seed(int(seed))
        self._initialization_generator.manual_seed(int(seed) + 456789123)
        return int(seed)

    def _extract_reset_mask(self, tensordict: TensorDictBase | None) -> torch.Tensor:
        if tensordict is None or "_reset" not in tensordict.keys():
            return torch.ones(
                self.num_envs, dtype=torch.bool, device=self.device
            )
        reset = tensordict.get("_reset").to(device=self.device, dtype=torch.bool)
        return reset.reshape(self.num_envs, -1).any(dim=-1)

    def _actor_tensordict(
        self,
        observation: torch.Tensor,
        action_mask: torch.Tensor,
        include_reward: torch.Tensor | None = None,
    ) -> TensorDict:
        metrics = self._state.episode_metrics[:, None, :].expand(
            -1, self.num_actors, -1
        )
        info = TensorDict(
            {
                name: metrics[..., index].unsqueeze(-1)
                for index, name in enumerate(METRIC_NAMES)
            },
            batch_size=torch.Size([self.num_envs, self.num_actors]),
            device=self.device,
        )
        data = {
            "observation": observation,
            "action_mask": action_mask,
            "info": info,
        }
        if include_reward is not None:
            data["reward"] = include_reward[:, None, :].expand(
                -1, self.num_actors, -1
            )
        return TensorDict(
            data,
            batch_size=torch.Size([self.num_envs, self.num_actors]),
            device=self.device,
        )

    def _base_output(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.dynamics.local_observation(self._state),
            self.dynamics.global_state(self._state),
            self.dynamics.action_mask(self._state),
        )

    def _reset(self, tensordict: TensorDictBase | None = None) -> TensorDict:
        reset_mask = self._extract_reset_mask(tensordict)
        self._state = self.dynamics.reset(
            self._state, reset_mask, self._initialization_generator
        )
        self._last_events = None
        observation, global_state, action_mask = self._base_output()
        zeros = torch.zeros(
            (self.num_envs, 1), dtype=torch.bool, device=self.device
        )
        return TensorDict(
            {
                self.group_name: self._actor_tensordict(
                    observation, action_mask
                ),
                "state": global_state,
                "done": zeros,
                "terminated": zeros.clone(),
                "truncated": zeros.clone(),
            },
            batch_size=self.batch_size,
            device=self.device,
        )

    def _step(self, tensordict: TensorDictBase) -> TensorDict:
        actions = tensordict.get((self.group_name, "action"))
        self._state, events = self.dynamics.step(
            self._state,
            actions,
            self._prediction_generator,
            self._initialization_generator,
        )
        self._last_events = events
        observation, global_state, action_mask = self._base_output()
        done = events.terminated | events.truncated
        return TensorDict(
            {
                self.group_name: self._actor_tensordict(
                    observation, action_mask, include_reward=events.reward
                ),
                "state": global_state,
                "done": done,
                "terminated": events.terminated,
                "truncated": events.truncated,
            },
            batch_size=self.batch_size,
            device=self.device,
        )

    def snapshot(self, env_index: int | None = None):
        """Copy one selected environment to an immutable CPU render snapshot."""
        from .rendering.snapshot import RenderSnapshot

        index = self.render_env_index if env_index is None else int(env_index)
        if not 0 <= index < self.num_envs:
            raise IndexError("env_index is outside the environment batch")
        actor_status = self.dynamics.actors.status(self._state.actor_phase)
        return RenderSnapshot.from_tensors(
            config=self.config,
            particle_state=self._state.particle_state[index],
            particle_class=self._state.particle_class[index],
            particle_spawned=self._state.particle_spawned[index],
            particle_active=self._state.particle_active[index],
            particle_outcome=self._state.particle_outcome[index],
            actor_status=actor_status[index],
            pending_delay=self._state.pending_delay[index],
            physics_step=self._state.physics_step[index],
        )

    def render(self):
        """Render only the configured batch member; training remains untouched."""
        from .rendering.renderer import MatplotlibRenderer

        if self._renderer is None:
            self._renderer = MatplotlibRenderer(self.config)
        frame = self._renderer.render(self.snapshot())
        if self.config.render_mode == "human":
            self._renderer.show(frame)
        return frame

    def close(self, *, raise_if_closed: bool = True) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        super().close(raise_if_closed=raise_if_closed)

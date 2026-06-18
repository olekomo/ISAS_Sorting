"""GPU-vectorized local observations, centralized state and action masks."""

from __future__ import annotations

import math

import torch
from torch import nn

from .actors import ActorDynamics
from .config import ISASConfig
from .state import SimulationState


class ObservationEncoder(nn.Module):
    """Build observations with the legacy six particle feature blocks.

    The existing feature order is retained:
    ``class, x, y, vx, vy, existence``. One additional block containing all
    normalized pending activation delays is appended because delayed actions
    add scheduler state that did not exist in the old binary interface.
    """

    particle_feature_dim = 6

    def __init__(
        self,
        config: ISASConfig,
        actor_dynamics: ActorDynamics,
        device: torch.device | str,
    ):
        super().__init__()
        self.config = config
        self.actor_dynamics = actor_dynamics
        self.K = config.obs_num_particles
        self.A = config.num_actors
        self.P = config.num_particles
        self.obs_dim = self.K * self.particle_feature_dim + 1 + 2 + 3 * self.A
        self.state_dim = self.K * self.particle_feature_dim + 2 * self.A

        actor_position = torch.tensor(
            config.grid.actor_positions, dtype=torch.float32, device=device
        )
        identity = torch.eye(self.A, dtype=torch.float32, device=device)
        self.register_buffer("actor_position", actor_position)
        self.register_buffer("identity", identity)

        self.x_mean = config.grid.array_end / 2.0
        self.x_scale = max(config.grid.array_end / math.sqrt(12.0), 1e-6)
        self.y_mean = config.area_width / 2.0
        self.y_scale = max(config.area_width / math.sqrt(12.0), 1e-6)
        self.vx_mean = config.v0_x
        self.vx_scale = max(config.particle_v0_x_stddev, 1e-6)
        self.vy_mean = config.particle_v0_y_mean
        self.vy_scale = max(config.particle_v0_y_stddev, 1e-6)
        self.t_mean = config.cycle_time / 2.0
        self.t_scale = max(config.cycle_time / math.sqrt(12.0), 1e-6)
        self.pending_scale = float(max(config.max_fire_delay_steps, 1))

    def _normalized_particle_features(
        self, state: SimulationState
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        active = state.particle_active
        cls = state.particle_class.to(torch.float32)
        x = (state.particle_state[..., 0] - self.x_mean) / self.x_scale
        vx = (state.particle_state[..., 1] - self.vx_mean) / self.vx_scale
        y = (state.particle_state[..., 2] - self.y_mean) / self.y_scale
        vy = (state.particle_state[..., 3] - self.vy_mean) / self.vy_scale
        existence = active.to(torch.float32)
        zero = torch.zeros((), dtype=torch.float32, device=active.device)
        return tuple(
            torch.where(active, feature, zero)
            for feature in (cls, x, y, vx, vy, existence)
        )  # type: ignore[return-value]

    def actor_time(self, state: SimulationState) -> torch.Tensor:
        elapsed = self.actor_dynamics.elapsed_time(state.actor_phase)
        if self.config.continuous_actor_states:
            return (elapsed - self.t_mean) / self.t_scale
        return elapsed / max(self.config.cycle_time, self.config.T)

    def pending_delay(self, state: SimulationState) -> torch.Tensor:
        return torch.where(
            state.pending_delay >= 0,
            state.pending_delay.to(torch.float32) / self.pending_scale,
            torch.full_like(state.pending_delay, -1, dtype=torch.float32),
        )

    def _gather(self, value: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        # value [B,P], indices [B,A,K]
        expanded = value[:, None, :].expand(-1, self.A, -1)
        return torch.gather(expanded, dim=-1, index=indices)

    def _topk_indices(self, score: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Pad if K > P while keeping a fixed output shape.
        if self.K <= self.P:
            values, indices = torch.topk(score, k=self.K, dim=-1, largest=False)
            return indices, torch.isfinite(values)
        padding = self.K - self.P
        pad_score = torch.full(
            (*score.shape[:-1], padding),
            torch.inf,
            device=score.device,
            dtype=score.dtype,
        )
        padded = torch.cat((score, pad_score), dim=-1)
        values, indices = torch.topk(padded, k=self.K, dim=-1, largest=False)
        valid = torch.isfinite(values) & (indices < self.P)
        return indices.clamp_max(self.P - 1), valid

    def local_observation(self, state: SimulationState) -> torch.Tensor:
        cls, x, y, vx, vy, existence = self._normalized_particle_features(state)
        active = state.particle_active

        physical_position = state.particle_state[..., (0, 2)]
        if self.config.agent_centric_obs:
            relative_physical = (
                physical_position[:, None, :, :]
                - self.actor_position[None, :, None, :]
            )
            score = (
                relative_physical[..., 0].square()
                + self.config.y_distance_weight
                * relative_physical[..., 1].square()
            )
        else:
            score = physical_position[..., 0][:, None, :].expand(
                -1, self.A, -1
            )
        score = score.masked_fill(~active[:, None, :], torch.inf)
        indices, valid = self._topk_indices(score)

        selected_cls = self._gather(cls, indices)
        selected_x = self._gather(x, indices)
        selected_y = self._gather(y, indices)
        selected_vx = self._gather(vx, indices)
        selected_vy = self._gather(vy, indices)
        selected_existence = self._gather(existence, indices)

        actor_x = (self.actor_position[:, 0] - self.x_mean) / self.x_scale
        actor_y = (self.actor_position[:, 1] - self.y_mean) / self.y_scale
        if self.config.agent_centric_obs:
            selected_x = selected_x - actor_x[None, :, None]
            selected_y = selected_y - actor_y[None, :, None]

        valid_f = valid.to(torch.float32)
        blocks = [
            selected_cls * valid_f,
            selected_x * valid_f,
            selected_y * valid_f,
            selected_vx * valid_f,
            selected_vy * valid_f,
            selected_existence * valid_f,
        ]
        particle_vector = torch.cat(blocks, dim=-1)

        timers = self.actor_time(state)
        pending = self.pending_delay(state)
        own_timer = timers.unsqueeze(-1)
        static_pos = torch.stack((actor_x, actor_y), dim=-1)[None, :, :].expand(
            state.batch_size, -1, -1
        )
        all_timers = timers[:, None, :].expand(-1, self.A, -1)
        all_pending = pending[:, None, :].expand(-1, self.A, -1)
        identity = self.identity[None, :, :].expand(state.batch_size, -1, -1)
        return torch.cat(
            (
                particle_vector,
                own_timer,
                static_pos,
                all_timers,
                all_pending,
                identity,
            ),
            dim=-1,
        )

    def global_state(self, state: SimulationState) -> torch.Tensor:
        cls, x, y, vx, vy, existence = self._normalized_particle_features(state)
        score = state.particle_state[..., 0].masked_fill(
            ~state.particle_active, torch.inf
        )
        score = score[:, None, :]
        indices, valid = self._topk_indices(score)
        indices = indices[:, 0, :]
        valid = valid[:, 0, :].to(torch.float32)

        def gather_global(value: torch.Tensor) -> torch.Tensor:
            return torch.gather(value, -1, indices) * valid

        particles = torch.cat(
            tuple(gather_global(value) for value in (cls, x, y, vx, vy, existence)),
            dim=-1,
        )
        return torch.cat(
            (particles, self.actor_time(state), self.pending_delay(state)), dim=-1
        )

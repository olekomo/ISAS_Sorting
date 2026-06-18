"""Tensor state containers used by the simulator core."""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch


@dataclass
class SimulationState:
    particle_state: torch.Tensor          # [B, P, 4] = x, vx, y, vy
    particle_initial_state: torch.Tensor  # [B, P, 4], sampled at reset
    particle_class: torch.Tensor          # [B, P], int64
    particle_spawned: torch.Tensor        # [B, P], bool
    particle_active: torch.Tensor         # [B, P], bool
    particle_resolved: torch.Tensor       # [B, P], bool
    particle_outcome: torch.Tensor        # [B, P], int64: 0/1/2/3
    potential_disturbances: torch.Tensor  # [B, A, P], bool

    actor_phase: torch.Tensor             # [B, A], -1 means READY
    pending_delay: torch.Tensor           # [B, A], -1 means no plan

    spawned_count: torch.Tensor           # [B], int64
    physics_step: torch.Tensor             # [B], int64
    decision_step: torch.Tensor            # [B], int64
    inactivity_count: torch.Tensor         # [B], int64
    episode_metrics: torch.Tensor          # [B, M], float32

    def clone(self) -> "SimulationState":
        return SimulationState(
            **{field.name: getattr(self, field.name).clone() for field in fields(self)}
        )

    @property
    def batch_size(self) -> int:
        return self.particle_state.shape[0]


@dataclass
class SimulationEvents:
    hit: torch.Tensor               # [B, P]
    disturbed: torch.Tensor         # [B, P]
    end_of_line: torch.Tensor       # [B, P]
    spawned: torch.Tensor           # [B, P]
    activated: torch.Tensor         # [B, A]
    invalid_action: torch.Tensor    # [B, A]
    reward: torch.Tensor            # [B, 1]
    terminated: torch.Tensor        # [B, 1]
    truncated: torch.Tensor         # [B, 1]

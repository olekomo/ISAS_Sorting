"""Immutable CPU snapshot passed from simulation to renderers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..config import ISASConfig


@dataclass(frozen=True)
class RenderSnapshot:
    actor_positions: np.ndarray
    actor_status: np.ndarray
    pending_delay: np.ndarray
    particle_state: np.ndarray
    particle_class: np.ndarray
    particle_spawned: np.ndarray
    particle_active: np.ndarray
    particle_outcome: np.ndarray
    physics_step: int
    array_end: float
    area_width: float
    actor_length: float
    actor_width: float
    particle_radius: float

    @classmethod
    def from_tensors(
        cls,
        *,
        config: ISASConfig,
        particle_state: torch.Tensor,
        particle_class: torch.Tensor,
        particle_spawned: torch.Tensor,
        particle_active: torch.Tensor,
        particle_outcome: torch.Tensor,
        actor_status: torch.Tensor,
        pending_delay: torch.Tensor,
        physics_step: torch.Tensor,
    ) -> "RenderSnapshot":
        def cpu(value: torch.Tensor) -> np.ndarray:
            return value.detach().cpu().numpy().copy()

        return cls(
            actor_positions=np.asarray(config.grid.actor_positions, dtype=np.float32),
            actor_status=cpu(actor_status),
            pending_delay=cpu(pending_delay),
            particle_state=cpu(particle_state),
            particle_class=cpu(particle_class),
            particle_spawned=cpu(particle_spawned),
            particle_active=cpu(particle_active),
            particle_outcome=cpu(particle_outcome),
            physics_step=int(physics_step.detach().cpu().item()),
            array_end=config.grid.array_end,
            area_width=config.area_width,
            actor_length=config.actor_length,
            actor_width=config.actor_width,
            particle_radius=config.particle_radius,
        )

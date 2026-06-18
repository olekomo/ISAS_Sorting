"""Batched deterministic particle-actuator contact model."""

from __future__ import annotations

import torch
from torch import nn

from .config import ISASConfig
from .enums import ActorStatus


class ContactEngine(nn.Module):
    """Vectorized equivalent of the legacy 2-D contact simulator."""

    def __init__(self, config: ISASConfig, device: torch.device | str):
        super().__init__()
        actor_position = torch.tensor(
            config.grid.actor_positions, dtype=torch.float32, device=device
        )
        actor_width = torch.full(
            (config.num_actors,), config.actor_width,
            dtype=torch.float32, device=device,
        )
        spatial_tolerance = torch.tensor(
            [config.v0_x * config.T / 2.0, 0.0],
            dtype=torch.float32,
            device=device,
        )
        disturbance_center = actor_position + torch.tensor(
            [config.dist_area_off_x, config.dist_area_off_y],
            dtype=torch.float32,
            device=device,
        )
        disturbance_half_extent = torch.tensor(
            [config.dist_area_len / 2.0, config.dist_area_width / 2.0],
            dtype=torch.float32,
            device=device,
        ).expand(config.num_actors, 2)

        self.register_buffer("actor_position", actor_position)
        self.register_buffer("actor_width", actor_width)
        self.register_buffer("spatial_tolerance", spatial_tolerance)
        self.register_buffer("disturbance_center", disturbance_center)
        self.register_buffer("disturbance_half_extent", disturbance_half_extent)

    def calculate(
        self,
        particle_state: torch.Tensor,
        particle_active: torch.Tensor,
        actor_status: torch.Tensor,
        potential_disturbances: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return hit, disturbed and updated potential-disturbance tensors."""
        position = particle_state[..., (0, 2)]  # [B, P, 2]
        relative_actor = position[:, None, :, :] - self.actor_position[None, :, None, :]
        hit_tolerance = self.spatial_tolerance[None, None, None, :].expand_as(relative_actor).clone()
        hit_tolerance[..., 1] = hit_tolerance[..., 1] + self.actor_width[None, :, None] / 2.0
        potential_hits = (
            (relative_actor.abs() <= hit_tolerance).all(dim=-1)
            & particle_active[:, None, :]
        )

        relative_disturbance = (
            position[:, None, :, :] - self.disturbance_center[None, :, None, :]
        )
        in_disturbance_area = (
            (
                relative_disturbance.abs()
                <= self.disturbance_half_extent[None, :, None, :]
            ).all(dim=-1)
            & particle_active[:, None, :]
        )

        is_hit = actor_status == int(ActorStatus.HIT)
        is_up_or_down = (
            (actor_status == int(ActorStatus.UP))
            | (actor_status == int(ActorStatus.DOWN))
        )
        is_disturbance_capable = is_hit | is_up_or_down

        hit_by_actor = potential_hits & is_hit.unsqueeze(-1)
        direct_disturbance = potential_hits & is_up_or_down.unsqueeze(-1)

        left_disturbance_without_same_actor_hit = (
            potential_disturbances
            & particle_active[:, None, :]
            & ~in_disturbance_area
            & ~hit_by_actor
        )

        disturbed_by_actor = direct_disturbance | left_disturbance_without_same_actor_hit
        disturbed = disturbed_by_actor.any(dim=1)
        hit = hit_by_actor.any(dim=1) & ~disturbed

        updated_potential = potential_disturbances | (
            in_disturbance_area & is_disturbance_capable.unsqueeze(-1)
        )
        return hit, disturbed, updated_potential

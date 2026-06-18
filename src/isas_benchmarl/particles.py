"""Vectorized constant-velocity particle dynamics with fixed particle slots."""

from __future__ import annotations

import torch
from torch import nn

from .config import ISASConfig


class ParticleDynamics(nn.Module):
    """Create and move all particles without changing tensor shapes.

    Slots are filled monotonically from ``0`` to ``num_particles - 1``. A slot
    is never removed; ``particle_active`` and ``particle_resolved`` describe its
    lifecycle. This is the central requirement for efficient batched GPU use.
    """

    def __init__(self, config: ISASConfig, device: torch.device | str):
        super().__init__()
        self.config = config
        self.num_particles = config.num_particles
        self.class_labels = tuple(config.particle_class_labels)
        self.num_processes = len(self.class_labels)

        T = config.T
        transition = torch.tensor(
            [[1.0, T, 0.0, 0.0],
             [0.0, 1.0, 0.0, 0.0],
             [0.0, 0.0, 1.0, T],
             [0.0, 0.0, 0.0, 1.0]],
            dtype=torch.float32,
            device=device,
        )
        self.register_buffer("transition", transition)

        q = float(config.simulation_noise)
        covariance = torch.zeros((4, 4), dtype=torch.float32, device=device)
        base = torch.tensor(
            [[T**3 / 3.0, T**2 / 2.0], [T**2 / 2.0, T]],
            dtype=torch.float32,
            device=device,
        )
        covariance[:2, :2] = q * base
        covariance[2:, 2:] = q * base
        if q > 0.0:
            # The analytical covariance is positive semidefinite. A tiny jitter
            # only protects float32 Cholesky at extreme parameter scales.
            noise_factor = torch.linalg.cholesky(
                covariance + torch.eye(4, device=device) * 1e-12
            )
        else:
            noise_factor = torch.zeros_like(covariance)
        self.register_buffer("noise_factor", noise_factor)
        self.register_buffer(
            "slot_indices", torch.arange(self.num_particles, device=device)
        )
        self.register_buffer(
            "process_labels",
            torch.tensor(self.class_labels, dtype=torch.int64, device=device),
        )

    @staticmethod
    def _normal(
        shape: tuple[int, ...],
        *,
        mean: float,
        std: float,
        device: torch.device,
        generator: torch.Generator,
    ) -> torch.Tensor:
        if std == 0.0:
            return torch.full(shape, mean, dtype=torch.float32, device=device)
        return torch.randn(
            shape, device=device, generator=generator, dtype=torch.float32
        ) * std + mean

    def move(
        self,
        particle_state: torch.Tensor,
        particle_active: torch.Tensor,
        generator: torch.Generator,
    ) -> torch.Tensor:
        """Advance active particles by one ``T`` interval and reflect at walls."""
        proposed = particle_state @ self.transition.T
        if self.config.simulation_noise > 0.0:
            epsilon = torch.randn(
                particle_state.shape,
                dtype=particle_state.dtype,
                device=particle_state.device,
                generator=generator,
            )
            proposed = proposed + epsilon @ self.noise_factor.T

        proposed = self._reflect_y(proposed)
        return torch.where(particle_active.unsqueeze(-1), proposed, particle_state)

    def _reflect_y(self, state: torch.Tensor) -> torch.Tensor:
        """Mirror positions and y velocity for arbitrary wall overshoot."""
        low = float(self.config.particle_radius)
        high = float(self.config.area_width - self.config.particle_radius)
        span = high - low

        y = state[..., 2]
        vy = state[..., 3]
        shifted = y - low
        period = 2.0 * span
        wrapped = torch.remainder(shifted, period)
        reflected_half = wrapped > span
        reflected_y = torch.where(reflected_half, period - wrapped, wrapped) + low

        # Each crossed half-period reverses direction. floor division works for
        # both positive and negative overshoots after using absolute segment id.
        segment = torch.floor(shifted / span).to(torch.int64)
        flip = torch.remainder(segment, 2) != 0
        reflected_vy = torch.where(flip, -vy, vy)

        result = state.clone()
        result[..., 2] = reflected_y
        result[..., 3] = reflected_vy
        return result

    def _sample_birth_counts(
        self,
        batch_size: int,
        remaining: torch.Tensor,
        generator: torch.Generator,
        device: torch.device,
    ) -> torch.Tensor:
        """Sample per-process births and cap them without replacement.

        The legacy setup gives each process half of ``default_birth_rate``.
        For a general number of configured class labels, the total rate is split
        evenly across processes.
        """
        mean = self.config.grid.default_birth_rate / self.num_processes
        std = self.config.particle_birth_rate_stddev
        raw = self._normal(
            (batch_size, self.num_processes),
            mean=mean,
            std=std,
            device=device,
            generator=generator,
        )
        requested = torch.round(raw).to(torch.int64).clamp_min_(0)
        total = requested.sum(-1)
        overflow = total > remaining

        # Exact multivariate-hypergeometric style capping: represent every
        # requested item as a token, assign random priorities, and retain the
        # first ``remaining`` tokens. Shapes stay static at C * P.
        token_rank = self.slot_indices.view(1, 1, -1)
        valid_tokens = token_rank < requested.unsqueeze(-1)
        priorities = torch.rand(
            (batch_size, self.num_processes, self.num_particles),
            device=device,
            generator=generator,
        )
        priorities = priorities.masked_fill(~valid_tokens, -1.0)
        selected = priorities.flatten(1).topk(
            k=self.num_particles, dim=-1, largest=True
        ).indices
        selected_process = selected // self.num_particles
        selected_rank = self.slot_indices.view(1, -1) < remaining.unsqueeze(-1)
        selected_counts = torch.nn.functional.one_hot(
            selected_process, num_classes=self.num_processes
        ).to(torch.int64)
        selected_counts = (
            selected_counts * selected_rank.unsqueeze(-1)
        ).sum(dim=1)
        return torch.where(overflow.unsqueeze(-1), selected_counts, requested)

    def sample_initial_states(
        self,
        shape: tuple[int, int],
        generator: torch.Generator,
        device: torch.device,
    ) -> torch.Tensor:
        """Sample the legacy truncated spawn distribution with bounded retries."""
        batch_size, num_particles = shape
        result = torch.zeros(
            (batch_size, num_particles, 4), dtype=torch.float32, device=device
        )
        valid = torch.zeros(shape, dtype=torch.bool, device=device)

        y_mean = self.config.area_width * self.config.particle_spawn_y_fraction
        y_std = self.config.particle_spawn_y_stddev_factor * self.config.area_width
        low_y = self.config.particle_radius
        high_y = self.config.area_width - self.config.particle_radius

        # The legacy code oversamples recursively. Fixed retries retain a static
        # graph; the final fallback clips only pathological residual samples.
        for _ in range(64):
            vx = self._normal(
                shape,
                mean=self.config.v0_x,
                std=self.config.particle_v0_x_stddev,
                device=device,
                generator=generator,
            )
            vy = self._normal(
                shape,
                mean=self.config.particle_v0_y_mean,
                std=self.config.particle_v0_y_stddev,
                device=device,
                generator=generator,
            )
            x = self._normal(
                shape,
                mean=self.config.particle_spawn_x,
                std=self.config.particle_spawn_x_stddev,
                device=device,
                generator=generator,
            )
            y = self._normal(
                shape,
                mean=y_mean,
                std=y_std,
                device=device,
                generator=generator,
            )
            candidate = torch.stack((x, vx, y, vy), dim=-1)
            candidate_valid = (
                (vx > 0.0)
                & (x <= 0.0)
                & (x >= -vx * self.config.T)
                & (y >= low_y)
                & (y <= high_y)
            )
            take = (~valid) & candidate_valid
            result = torch.where(take.unsqueeze(-1), candidate, result)
            valid = valid | candidate_valid

        # Always construct the fallback to keep the execution graph free of
        # data-dependent Python branches. It is selected only for residual
        # invalid samples.
        vx = self._normal(
            shape,
            mean=self.config.v0_x,
            std=self.config.particle_v0_x_stddev,
            device=device,
            generator=generator,
        ).clamp_min(1e-6)
        vy = self._normal(
            shape,
            mean=self.config.particle_v0_y_mean,
            std=self.config.particle_v0_y_stddev,
            device=device,
            generator=generator,
        ).clamp(
            self.config.particle_velocity_y_min,
            self.config.particle_velocity_y_max,
        )
        x = self._normal(
            shape,
            mean=self.config.particle_spawn_x,
            std=self.config.particle_spawn_x_stddev,
            device=device,
            generator=generator,
        )
        x = torch.maximum(x, -vx * self.config.T).clamp_max(0.0)
        y = self._normal(
            shape,
            mean=y_mean,
            std=y_std,
            device=device,
            generator=generator,
        ).clamp(low_y, high_y)
        fallback = torch.stack((x, vx, y, vy), dim=-1)
        result = torch.where((~valid).unsqueeze(-1), fallback, result)
        return result

    def spawn(
        self,
        particle_state: torch.Tensor,
        particle_initial_state: torch.Tensor,
        particle_class: torch.Tensor,
        particle_spawned: torch.Tensor,
        particle_active: torch.Tensor,
        spawned_count: torch.Tensor,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Spawn a sampled number of particles into the next free fixed slots."""
        batch_size = particle_state.shape[0]
        device = particle_state.device
        remaining = self.num_particles - spawned_count
        counts = self._sample_birth_counts(
            batch_size, remaining, generator, device
        )

        spawned_now = torch.zeros_like(particle_spawned)
        start = spawned_count

        for process_index, class_label in enumerate(self.class_labels):
            count = counts[:, process_index]
            end = start + count
            process_slots = (
                (self.slot_indices.unsqueeze(0) >= start.unsqueeze(-1))
                & (self.slot_indices.unsqueeze(0) < end.unsqueeze(-1))
            )
            spawned_now = spawned_now | process_slots
            particle_class = torch.where(
                process_slots,
                torch.full_like(particle_class, int(class_label)),
                particle_class,
            )
            start = end

        particle_state = torch.where(
            spawned_now.unsqueeze(-1), particle_initial_state, particle_state
        )
        particle_spawned = particle_spawned | spawned_now
        particle_active = particle_active | spawned_now
        spawned_count = spawned_count + counts.sum(-1)
        return (
            particle_state,
            particle_class,
            particle_spawned,
            particle_active,
            spawned_count,
            spawned_now,
        )

"""Framework-independent batched PyTorch simulation core."""

from __future__ import annotations

from dataclasses import fields

import torch
from torch import nn

from .actors import ActorDynamics
from .config import ISASConfig
from .contacts import ContactEngine
from .observations import ObservationEncoder
from .particles import ParticleDynamics
from .rewards import RewardEngine
from .state import SimulationEvents, SimulationState


class ISASDynamics(nn.Module):
    """Complete vectorized simulator, independent from TorchRL and BenchMARL."""

    def __init__(self, config: ISASConfig, device: torch.device | str):
        super().__init__()
        self.config = config
        self.device_ref = torch.device(device)
        self.actors = ActorDynamics(config, device)
        self.particles = ParticleDynamics(config, device)
        self.contacts = ContactEngine(config, device)
        self.rewards = RewardEngine(config)
        self.observations = ObservationEncoder(config, self.actors, device)

    def empty_state(self, batch_size: int) -> SimulationState:
        device = self.device_ref
        B, P, A = batch_size, self.config.num_particles, self.config.num_actors
        return SimulationState(
            particle_state=torch.zeros((B, P, 4), dtype=torch.float32, device=device),
            particle_initial_state=torch.zeros((B, P, 4), dtype=torch.float32, device=device),
            particle_class=torch.zeros((B, P), dtype=torch.int64, device=device),
            particle_spawned=torch.zeros((B, P), dtype=torch.bool, device=device),
            particle_active=torch.zeros((B, P), dtype=torch.bool, device=device),
            particle_resolved=torch.zeros((B, P), dtype=torch.bool, device=device),
            particle_outcome=torch.zeros((B, P), dtype=torch.int64, device=device),
            potential_disturbances=torch.zeros(
                (B, A, P), dtype=torch.bool, device=device
            ),
            actor_phase=torch.full((B, A), -1, dtype=torch.int64, device=device),
            pending_delay=torch.full((B, A), -1, dtype=torch.int64, device=device),
            spawned_count=torch.zeros((B,), dtype=torch.int64, device=device),
            physics_step=torch.zeros((B,), dtype=torch.int64, device=device),
            decision_step=torch.zeros((B,), dtype=torch.int64, device=device),
            inactivity_count=torch.zeros((B,), dtype=torch.int64, device=device),
            episode_metrics=torch.zeros(
                (B, self.rewards.num_metrics), dtype=torch.float32, device=device
            ),
        )

    def reset(
        self,
        state: SimulationState | None,
        reset_mask: torch.Tensor | None,
        initialization_generator: torch.Generator,
    ) -> SimulationState:
        """Reset all or selected environments without changing batch shape."""
        if state is None:
            if reset_mask is None:
                raise ValueError("batch size is required when state is None")
            batch_size = int(reset_mask.numel())
            fresh = self.empty_state(batch_size)
            fresh.particle_initial_state = self.particles.sample_initial_states(
                (batch_size, self.config.num_particles),
                initialization_generator,
                self.device_ref,
            )
            return fresh

        mask = (
            torch.ones(state.batch_size, dtype=torch.bool, device=self.device_ref)
            if reset_mask is None
            else reset_mask.to(device=self.device_ref, dtype=torch.bool).reshape(
                state.batch_size
            )
        )
        fresh = self.empty_state(state.batch_size)
        fresh.particle_initial_state = self.particles.sample_initial_states(
            (state.batch_size, self.config.num_particles),
            initialization_generator,
            self.device_ref,
        )
        values: dict[str, torch.Tensor] = {}
        for field in fields(state):
            old_value = getattr(state, field.name)
            new_value = getattr(fresh, field.name)
            expanded_mask = mask.view(
                state.batch_size, *([1] * (old_value.ndim - 1))
            )
            values[field.name] = torch.where(expanded_mask, new_value, old_value)
        return SimulationState(**values)

    def action_mask(self, state: SimulationState) -> torch.Tensor:
        return self.actors.action_mask(state.actor_phase, state.pending_delay)

    def local_observation(self, state: SimulationState) -> torch.Tensor:
        return self.observations.local_observation(state)

    def global_state(self, state: SimulationState) -> torch.Tensor:
        return self.observations.global_state(state)

    def step(
        self,
        state: SimulationState,
        actions: torch.Tensor,
        prediction_generator: torch.Generator,
        initialization_generator: torch.Generator,
    ) -> tuple[SimulationState, SimulationEvents]:
        """Apply one policy action and simulate ``frame_factor`` physics steps."""
        pending_delay, invalid_action = self.actors.schedule(
            state.actor_phase, state.pending_delay, actions
        )
        state.pending_delay = pending_delay

        B, P, A = (
            state.batch_size,
            self.config.num_particles,
            self.config.num_actors,
        )
        reward_total = torch.zeros((B, 1), dtype=torch.float32, device=self.device_ref)
        hit_total = torch.zeros((B, P), dtype=torch.bool, device=self.device_ref)
        disturbed_total = torch.zeros_like(hit_total)
        end_total = torch.zeros_like(hit_total)
        spawned_total = torch.zeros_like(hit_total)
        activated_total = torch.zeros((B, A), dtype=torch.bool, device=self.device_ref)

        for substep in range(self.config.frame_factor):
            # Legacy order: move existing particles, update actors, then create
            # particles. Newly created particles therefore first move next step.
            state.particle_state = self.particles.move(
                state.particle_state, state.particle_active, prediction_generator
            )
            (
                state.actor_phase,
                state.pending_delay,
                activated,
            ) = self.actors.physics_step(state.actor_phase, state.pending_delay)

            (
                state.particle_state,
                state.particle_class,
                state.particle_spawned,
                state.particle_active,
                state.spawned_count,
                spawned_now,
            ) = self.particles.spawn(
                state.particle_state,
                state.particle_initial_state,
                state.particle_class,
                state.particle_spawned,
                state.particle_active,
                state.spawned_count,
                initialization_generator,
            )

            actor_status = self.actors.status(state.actor_phase)
            hit, disturbed, state.potential_disturbances = self.contacts.calculate(
                state.particle_state,
                state.particle_active,
                actor_status,
                state.potential_disturbances,
            )
            end_of_line = (
                state.particle_active
                & (state.particle_state[..., 0] >= self.config.grid.array_end)
                & ~hit
                & ~disturbed
            )
            resolved_now = hit | disturbed | end_of_line

            state.particle_outcome = torch.where(
                hit, torch.ones_like(state.particle_outcome), state.particle_outcome
            )
            state.particle_outcome = torch.where(
                disturbed,
                torch.full_like(state.particle_outcome, 2),
                state.particle_outcome,
            )
            state.particle_outcome = torch.where(
                end_of_line,
                torch.full_like(state.particle_outcome, 3),
                state.particle_outcome,
            )
            state.particle_resolved = state.particle_resolved | resolved_now
            state.particle_active = state.particle_active & ~resolved_now
            state.potential_disturbances = (
                state.potential_disturbances
                & state.particle_active[:, None, :]
            )

            invalid_for_reward = (
                invalid_action
                if substep == 0
                else torch.zeros_like(invalid_action)
            )
            reward_result = self.rewards.calculate(
                state.particle_class,
                hit,
                disturbed,
                end_of_line,
                spawned_now,
                invalid_for_reward,
            )
            reward_total = reward_total + reward_result.reward
            state.episode_metrics = (
                state.episode_metrics + reward_result.metric_delta
            )

            any_active = state.particle_active.any(dim=-1)
            state.inactivity_count = torch.where(
                any_active,
                torch.zeros_like(state.inactivity_count),
                state.inactivity_count + 1,
            )
            state.physics_step = state.physics_step + 1

            hit_total = hit_total | hit
            disturbed_total = disturbed_total | disturbed
            end_total = end_total | end_of_line
            spawned_total = spawned_total | spawned_now
            activated_total = activated_total | activated

        state.decision_step = state.decision_step + 1
        all_spawned = state.spawned_count >= self.config.num_particles
        no_active = ~state.particle_active.any(dim=-1)
        terminated = (all_spawned & no_active).unsqueeze(-1)
        truncated = (
            (state.decision_step >= self.config.max_decision_steps)
            | (state.inactivity_count >= self.config.time_out)
        ).unsqueeze(-1) & ~terminated

        events = SimulationEvents(
            hit=hit_total,
            disturbed=disturbed_total,
            end_of_line=end_total,
            spawned=spawned_total,
            activated=activated_total,
            invalid_action=invalid_action,
            reward=reward_total,
            terminated=terminated,
            truncated=truncated,
        )
        return state, events

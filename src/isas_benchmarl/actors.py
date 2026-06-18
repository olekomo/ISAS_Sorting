"""Vectorized actuator state machine and delayed activation scheduler."""

from __future__ import annotations

import torch
from torch import nn

from .config import ISASConfig
from .enums import ActorStatus


class ActorDynamics(nn.Module):
    """State machine for all actors in all environments.

    Action semantics:
        0: do not create a new activation plan
        1: activate in the current physics step
        n>=2: activate after ``n-1`` physics steps

    An actor accepts a plan only while READY and when no plan is pending. A
    pending plan survives across decision boundaries. This avoids ambiguous
    replacement/queue semantics and produces a well-defined action mask.
    """

    def __init__(self, config: ISASConfig, device: torch.device | str):
        super().__init__()
        self.config = config
        phase_steps = torch.tensor(
            config.actor_phase_steps, dtype=torch.int64, device=device
        )
        self.register_buffer("phase_steps", phase_steps)
        self.register_buffer("phase_ends", phase_steps.cumsum(0))
        self.total_steps = int(phase_steps.sum().item())
        if self.total_steps <= 0:
            raise ValueError("The actor cycle must contain at least one physics step.")

    @staticmethod
    def ready(actor_phase: torch.Tensor) -> torch.Tensor:
        return actor_phase < 0

    @staticmethod
    def schedulable(
        actor_phase: torch.Tensor, pending_delay: torch.Tensor
    ) -> torch.Tensor:
        return (actor_phase < 0) & (pending_delay < 0)

    def action_mask(
        self, actor_phase: torch.Tensor, pending_delay: torch.Tensor
    ) -> torch.Tensor:
        schedulable = self.schedulable(actor_phase, pending_delay)
        shape = (*actor_phase.shape, self.config.num_actions)
        mask = torch.zeros(shape, dtype=torch.bool, device=actor_phase.device)
        mask[..., 0] = True
        mask[..., 1:] = schedulable.unsqueeze(-1)
        return mask

    def schedule(
        self,
        actor_phase: torch.Tensor,
        pending_delay: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if actions.ndim == actor_phase.ndim + 1 and actions.shape[-1] == 1:
            actions = actions.squeeze(-1)
        actions = actions.to(dtype=torch.int64)
        if actions.shape != actor_phase.shape:
            raise ValueError(
                f"Expected actions shape {tuple(actor_phase.shape)} or "
                f"{tuple(actor_phase.shape) + (1,)}, got {tuple(actions.shape)}."
            )
        in_range = (actions >= 0) & (actions < self.config.num_actions)
        wants_plan = (actions > 0) & in_range
        can_plan = self.schedulable(actor_phase, pending_delay)
        accepted = wants_plan & can_plan
        invalid = ~in_range | (wants_plan & ~can_plan)
        new_delay = (actions - 1).clamp(0, self.config.max_fire_delay_steps)
        pending_delay = torch.where(accepted, new_delay, pending_delay)
        return pending_delay, invalid

    def physics_step(
        self, actor_phase: torch.Tensor, pending_delay: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Advance one physics interval and execute due activation plans."""
        ready_before = self.ready(actor_phase)
        activate_now = (pending_delay == 0) & ready_before

        # Existing cycles advance first. Actors that finish become READY.
        was_busy = actor_phase >= 0
        advanced_phase = actor_phase + 1
        actor_phase = torch.where(was_busy, advanced_phase, actor_phase)
        actor_phase = torch.where(
            actor_phase >= self.total_steps,
            torch.full_like(actor_phase, -1),
            actor_phase,
        )

        # A due plan starts a fresh cycle in phase 0 in this same physics step.
        actor_phase = torch.where(activate_now, torch.zeros_like(actor_phase), actor_phase)

        # Pending delays count physics transitions. A plan with delay 1 becomes
        # due on the next call, not on the current call.
        pending_delay = torch.where(
            pending_delay > 0, pending_delay - 1, pending_delay
        )
        pending_delay = torch.where(
            activate_now, torch.full_like(pending_delay, -1), pending_delay
        )
        return actor_phase, pending_delay, activate_now

    def status(self, actor_phase: torch.Tensor) -> torch.Tensor:
        status = torch.full_like(actor_phase, int(ActorStatus.READY))
        busy = actor_phase >= 0
        ends = self.phase_ends

        status = torch.where(
            busy & (actor_phase < ends[0]),
            torch.full_like(status, int(ActorStatus.ACTIVATE)),
            status,
        )
        status = torch.where(
            busy & (actor_phase >= ends[0]) & (actor_phase < ends[1]),
            torch.full_like(status, int(ActorStatus.UP)),
            status,
        )
        status = torch.where(
            busy & (actor_phase >= ends[1]) & (actor_phase < ends[2]),
            torch.full_like(status, int(ActorStatus.HIT)),
            status,
        )
        status = torch.where(
            busy & (actor_phase >= ends[2]) & (actor_phase < ends[3]),
            torch.full_like(status, int(ActorStatus.DOWN)),
            status,
        )
        status = torch.where(
            busy & (actor_phase >= ends[3]) & (actor_phase < ends[4]),
            torch.full_like(status, int(ActorStatus.RESET)),
            status,
        )
        return status

    def elapsed_time(self, actor_phase: torch.Tensor) -> torch.Tensor:
        # Legacy t_act starts at zero in READY and grows in T-sized intervals.
        return torch.where(
            actor_phase >= 0,
            (actor_phase.to(torch.float32) + 1.0) * self.config.T,
            torch.zeros_like(actor_phase, dtype=torch.float32),
        )

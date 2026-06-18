"""Reward and metric calculation with legacy-compatible coefficients."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import ISASConfig


METRIC_NAMES = (
    "total_tns",
    "total_fns",
    "total_disturbed_neg",
    "total_disturbed_pos",
    "total_tps",
    "total_fps",
    "count_class_keep",
    "count_class_eject",
    "invalid_actions",
)


@dataclass(frozen=True)
class RewardResult:
    reward: torch.Tensor
    metric_delta: torch.Tensor


class RewardEngine:
    """Preserve the existing shared scalar reward while tensorizing it."""

    num_metrics = len(METRIC_NAMES)

    def __init__(self, config: ISASConfig):
        self.config = config

    def calculate(
        self,
        particle_class: torch.Tensor,
        hit: torch.Tensor,
        disturbed: torch.Tensor,
        end_of_line: torch.Tensor,
        spawned_now: torch.Tensor,
        invalid_action: torch.Tensor,
    ) -> RewardResult:
        eject_class = particle_class == 1
        keep_class = particle_class == 0

        legacy_tns = (eject_class & hit).sum(-1)
        legacy_fns = (keep_class & hit).sum(-1)
        legacy_dist_neg = (eject_class & disturbed).sum(-1)
        legacy_dist_pos = (keep_class & disturbed).sum(-1)
        legacy_tps = (keep_class & end_of_line).sum(-1)
        legacy_fps = (eject_class & end_of_line).sum(-1)
        spawned_keep = (keep_class & spawned_now).sum(-1)
        spawned_eject = (eject_class & spawned_now).sum(-1)
        invalid_count = invalid_action.sum(-1)

        reward = (
            -self.config.reward_costs.accept_cost * legacy_fns.to(torch.float32)
            + self.config.reward_costs.reject_cost * legacy_tns.to(torch.float32)
            + self.config.reward_costs.disturbed_cost
            * (legacy_dist_neg - legacy_dist_pos).to(torch.float32)
        ).unsqueeze(-1)

        metric_delta = torch.stack(
            (
                legacy_tns,
                legacy_fns,
                legacy_dist_neg,
                legacy_dist_pos,
                legacy_tps,
                legacy_fps,
                spawned_keep,
                spawned_eject,
                invalid_count,
            ),
            dim=-1,
        ).to(torch.float32)
        return RewardResult(reward=reward, metric_delta=metric_delta)

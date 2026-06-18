from dataclasses import replace

import torch

from isas_benchmarl import ISASConfig
from isas_benchmarl.actors import ActorDynamics


def test_frame_factor_consumes_physics_delay_steps(config: ISASConfig):
    cfg = replace(config, T=0.1, frame_factor=5, max_fire_delay_steps=8)
    actors = ActorDynamics(cfg, "cpu")
    phase = torch.full((1, cfg.num_actors), -1, dtype=torch.int64)
    pending = torch.full_like(phase, -1)

    actions = torch.zeros_like(phase)
    actions[0, 0] = 3  # activation after two complete physics intervals
    pending, invalid = actors.schedule(phase, pending, actions)
    assert not invalid.any()

    activated_at = []
    for physics_index in range(cfg.frame_factor):
        phase, pending, activated = actors.physics_step(phase, pending)
        if activated[0, 0]:
            activated_at.append(physics_index)

    assert activated_at == [2]
    assert phase[0, 0] == 2  # two further physics intervals elapsed after activation


def test_delay_can_span_multiple_policy_decisions(config: ISASConfig):
    cfg = replace(config, T=0.1, frame_factor=5, max_fire_delay_steps=8)
    actors = ActorDynamics(cfg, "cpu")
    phase = torch.full((1, cfg.num_actors), -1, dtype=torch.int64)
    pending = torch.full_like(phase, -1)

    actions = torch.zeros_like(phase)
    actions[0, 0] = 7  # six physics intervals
    pending, _ = actors.schedule(phase, pending, actions)

    for _ in range(cfg.frame_factor):
        phase, pending, activated = actors.physics_step(phase, pending)
        assert not activated[0, 0]
    assert pending[0, 0] == 1

    # A new decision arrives, but the pending plan remains authoritative.
    pending, invalid = actors.schedule(phase, pending, torch.zeros_like(actions))
    assert not invalid.any()
    phase, pending, activated = actors.physics_step(phase, pending)
    assert not activated[0, 0]
    phase, pending, activated = actors.physics_step(phase, pending)
    assert activated[0, 0]

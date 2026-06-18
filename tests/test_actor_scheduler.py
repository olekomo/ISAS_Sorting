import torch

from isas_benchmarl.actors import ActorDynamics
from isas_benchmarl.enums import ActorStatus


def test_delayed_action_semantics(config):
    actors = ActorDynamics(config, "cpu")
    phase = torch.full((1, config.num_actors), -1, dtype=torch.int64)
    pending = torch.full_like(phase, -1)

    actions = torch.zeros_like(phase)
    actions[0, 0] = 1  # now
    actions[0, 1] = 2  # after one physics step
    pending, invalid = actors.schedule(phase, pending, actions)
    assert not invalid.any()

    phase, pending, activated = actors.physics_step(phase, pending)
    assert activated[0, 0]
    assert not activated[0, 1]
    assert pending[0, 1] == 0
    assert actors.status(phase)[0, 0] == int(ActorStatus.ACTIVATE)

    phase, pending, activated = actors.physics_step(phase, pending)
    assert activated[0, 1]
    assert pending[0, 1] == -1


def test_pending_plan_masks_all_non_noop_actions(config):
    actors = ActorDynamics(config, "cpu")
    phase = torch.full((1, config.num_actors), -1, dtype=torch.int64)
    pending = torch.full_like(phase, -1)
    actions = torch.zeros_like(phase)
    actions[0, 0] = config.num_actions - 1
    pending, _ = actors.schedule(phase, pending, actions)
    mask = actors.action_mask(phase, pending)
    assert mask[0, 0, 0]
    assert not mask[0, 0, 1:].any()
    assert mask[0, 1].all()

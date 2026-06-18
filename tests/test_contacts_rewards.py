import torch

from isas_benchmarl.contacts import ContactEngine
from isas_benchmarl.enums import ActorStatus
from isas_benchmarl.rewards import RewardEngine


def test_contact_in_hit_window(config):
    contact = ContactEngine(config, "cpu")
    B, P, A = 1, config.num_particles, config.num_actors
    particle_state = torch.zeros((B, P, 4))
    particle_state[0, 0, 0] = contact.actor_position[0, 0]
    particle_state[0, 0, 2] = contact.actor_position[0, 1]
    active = torch.zeros((B, P), dtype=torch.bool)
    active[0, 0] = True
    status = torch.full((B, A), int(ActorStatus.READY), dtype=torch.int64)
    status[0, 0] = int(ActorStatus.HIT)
    potentials = torch.zeros((B, A, P), dtype=torch.bool)
    hit, disturbed, _ = contact.calculate(
        particle_state, active, status, potentials
    )
    assert hit[0, 0]
    assert not disturbed[0, 0]


def test_reward_formula_matches_legacy(config):
    reward = RewardEngine(config)
    classes = torch.tensor([[1, 0, 1, 0]])
    hit = torch.tensor([[True, True, False, False]])
    disturbed = torch.tensor([[False, False, True, True]])
    end = torch.zeros_like(hit)
    spawned = torch.zeros_like(hit)
    invalid = torch.zeros((1, config.num_actors), dtype=torch.bool)
    result = reward.calculate(classes, hit, disturbed, end, spawned, invalid)
    expected = (
        config.reward_costs.reject_cost
        - config.reward_costs.accept_cost
        + config.reward_costs.disturbed_cost * (1 - 1)
    )
    assert torch.allclose(result.reward, torch.tensor([[expected]]))

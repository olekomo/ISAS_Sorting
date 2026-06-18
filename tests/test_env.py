import torch
from tensordict import TensorDict
from torchrl.envs.utils import check_env_specs

from isas_benchmarl import ISASTorchEnv


def test_torchrl_specs_and_rollout(config):
    env = ISASTorchEnv(config, num_envs=4, device="cpu")
    check_env_specs(env)
    rollout = env.rollout(4)
    assert rollout.batch_size == torch.Size([4, 4])
    assert rollout["actors", "observation"].shape == (
        4,
        4,
        config.num_actors,
        env.observation_dim,
    )


def test_shapes_never_change(config):
    env = ISASTorchEnv(config, num_envs=3, device="cpu")
    td = env.reset()
    expected_particle_shape = (3, config.num_particles, 4)
    for _ in range(20):
        action = torch.zeros((3, config.num_actors), dtype=torch.int64)
        td.set(("actors", "action"), action)
        td = env.step(td)["next"]
        assert env._state.particle_state.shape == expected_particle_shape
        assert env._state.particle_active.shape == (3, config.num_particles)
        assert td["actors", "action_mask"].shape == (
            3,
            config.num_actors,
            config.num_actions,
        )


def test_partial_reset_only_resets_selected_batch_members(config):
    env = ISASTorchEnv(config, num_envs=3, device="cpu")
    td = env.reset()
    for _ in range(4):
        td.set(
            ("actors", "action"),
            torch.zeros((3, config.num_actors), dtype=torch.int64),
        )
        td = env.step(td)["next"]

    before = env._state.clone()
    reset_td = TensorDict(
        {"_reset": torch.tensor([[False], [True], [False]])},
        batch_size=[3],
    )
    env.reset(reset_td)
    assert env._state.decision_step[1] == 0
    assert env._state.physics_step[1] == 0
    assert torch.equal(env._state.decision_step[[0, 2]], before.decision_step[[0, 2]])
    assert torch.equal(env._state.particle_state[[0, 2]], before.particle_state[[0, 2]])

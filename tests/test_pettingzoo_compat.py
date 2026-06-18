from isas_benchmarl.compat.pettingzoo import ISASSortingEnv


def test_b1_pettingzoo_adapter(config):
    env = ISASSortingEnv(config)
    observations, infos = env.reset(seed=17)
    assert set(observations) == set(env.possible_agents)
    assert infos[env.possible_agents[0]]["action_mask"].shape == (
        config.num_actions,
    )

    actions = {agent: 0 for agent in env.agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)
    assert set(rewards) == set(env.possible_agents)
    assert not any(terminations.values())
    assert not any(truncations.values())
    assert env.state().shape == (env.torch_env.state_dim,)
    env.close()

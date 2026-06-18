from isas_benchmarl.benchmarl import ISASClass


def test_benchmarl_task_exposes_clean_specs(config):
    task = ISASClass("SORTING", config.__dict__)
    # Use the typed object directly because nested dataclass __dict__ is not a
    # serialized grid mapping. This also verifies the compatibility loader.
    task.config = config
    env = task.get_env_fun(2, False, config.seed, "cpu")()
    observation_spec = task.observation_spec(env)
    assert list(observation_spec["actors"].keys()) == ["observation"]
    assert list(task.action_spec(env)["actors"].keys()) == ["action"]
    assert list(task.action_mask_spec(env)["actors"].keys()) == ["action_mask"]
    assert list(task.state_spec(env).keys()) == ["state"]

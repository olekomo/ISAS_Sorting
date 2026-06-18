from isas_benchmarl import ISASTorchEnv
from isas_benchmarl.rendering import MatplotlibRenderer


def test_renderer_is_independent_from_step(config):
    env = ISASTorchEnv(config, num_envs=2, device="cpu")
    env.reset()
    before = env._state.clone()
    frame = MatplotlibRenderer(config).render(env.snapshot(env_index=1))
    assert frame.ndim == 3 and frame.shape[-1] == 3
    assert frame.dtype.name == "uint8"
    assert (env._state.particle_state == before.particle_state).all()

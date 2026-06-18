"""Validate specs and run a short fully batched rollout."""

from pathlib import Path

from torchrl.envs.utils import check_env_specs

from isas_benchmarl import ISASConfig, ISASTorchEnv


config = ISASConfig.from_yaml(Path("configs/isas_3x3.yaml"))
env = ISASTorchEnv(config, num_envs=16, device="cpu")
check_env_specs(env)
rollout = env.rollout(max_steps=8)
print("rollout batch:", rollout.batch_size)
print("observation shape:", rollout["actors", "observation"].shape)

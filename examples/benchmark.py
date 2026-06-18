"""Measure native environment throughput without model/training overhead."""

from argparse import ArgumentParser
from pathlib import Path
from time import perf_counter

import torch

from isas_benchmarl import ISASConfig, ISASTorchEnv


parser = ArgumentParser()
parser.add_argument("--config", default="configs/isas_3x3.yaml")
parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
parser.add_argument("--num-envs", type=int, default=1024)
parser.add_argument("--steps", type=int, default=1000)
args = parser.parse_args()

config = ISASConfig.from_yaml(Path(args.config))
env = ISASTorchEnv(config, num_envs=args.num_envs, device=args.device)
td = env.reset()

for _ in range(20):
    td.update(env.action_spec.rand())
    td = env.step(td)["next"]

if str(args.device).startswith("cuda"):
    torch.cuda.synchronize()
start = perf_counter()
for _ in range(args.steps):
    # no-op is a valid baseline that isolates physics and observations
    td.set(
        ("actors", "action"),
        torch.zeros(
            (args.num_envs, config.num_actors),
            dtype=torch.int64,
            device=args.device,
        ),
    )
    td = env.step(td)["next"]
if str(args.device).startswith("cuda"):
    torch.cuda.synchronize()
elapsed = perf_counter() - start
print(f"{args.num_envs * args.steps / elapsed:,.0f} env decisions/s")
print(f"{args.num_envs * args.steps * config.frame_factor / elapsed:,.0f} physics steps/s")

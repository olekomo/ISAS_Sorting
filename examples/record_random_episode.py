"""Generate an MP4 and a PNG independently from training."""

from pathlib import Path

import torch

from isas_benchmarl import ISASConfig, ISASTorchEnv
from isas_benchmarl.rendering import FrameRecorder, MatplotlibRenderer


config = ISASConfig.from_yaml(Path("configs/isas_3x3.yaml"))
env = ISASTorchEnv(config, num_envs=1, device="cpu", seed=config.seed)
renderer = MatplotlibRenderer(config)
recorder = FrameRecorder("artifacts/render")

td = env.reset()
frames = [renderer.render(env.snapshot())]
for _ in range(config.max_decision_steps):
    mask = td["actors", "action_mask"]
    probabilities = mask.to(torch.float32).reshape(-1, config.num_actions)
    action = torch.multinomial(probabilities, 1).reshape(1, config.num_actors)
    td.set(("actors", "action"), action)
    td = env.step(td)["next"]
    frames.append(renderer.render(env.snapshot()))
    if td["done"].all():
        break

print(recorder.save_image(frames[-1], "final.png"))
print(recorder.save_video(frames, "random_episode.mp4", fps=10))
env.close()

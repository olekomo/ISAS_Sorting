"""Render a deterministic episode from a BenchMARL experiment checkpoint."""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from benchmarl.experiment import Experiment
from torchrl.envs.utils import ExplorationType, set_exploration_type

from isas_benchmarl.rendering import FrameRecorder


def build_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--output-dir", default="artifacts/final_evaluation")
    parser.add_argument("--video-name", default="learned_episode.mp4")
    parser.add_argument("--image-name", default="learned_final.png")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Defaults to the task's configured maximum episode length.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    output_dir = Path(args.output_dir).expanduser().resolve()
    recorder = FrameRecorder(output_dir)

    experiment = Experiment.reload_from_file(
        str(checkpoint),
        experiment_patch={
            "sampling_device": args.device,
            "train_device": args.device,
            "buffer_device": args.device,
            "evaluation_episodes": 1,
            "evaluation_deterministic_actions": True,
            "render": False,
            "loggers": [],
            "create_json": False,
        },
    )

    frames: list[np.ndarray] = []

    def capture_frame(env, _td) -> None:
        frame = env.render()
        if isinstance(frame, torch.Tensor):
            frame = frame.detach().cpu().numpy()
        frames.append(np.asarray(frame))

    try:
        max_steps = args.max_steps or experiment.max_steps
        with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
            experiment.test_env.rollout(
                max_steps=max_steps,
                policy=experiment.policy,
                callback=capture_frame,
                auto_cast_to_device=True,
                break_when_any_done=True,
            )

        if not frames:
            raise RuntimeError("The evaluation produced no render frames.")

        video_path = recorder.save_video(
            frames,
            name=args.video_name,
            fps=args.fps,
        )
        image_path = recorder.save_image(frames[-1], name=args.image_name)

        print(f"Checkpoint: {checkpoint}")
        print(f"Captured frames: {len(frames)}")
        print(f"Video: {video_path.resolve()}")
        print(f"Final image: {image_path.resolve()}")
    finally:
        experiment.close()


if __name__ == "__main__":
    main()

"""Programmatic BenchMARL entry point with logging and checkpoint controls."""

from __future__ import annotations

from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path

import torch
import yaml
from benchmarl.algorithms import MappoConfig, MasacConfig
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models import MlpConfig

from isas_benchmarl.benchmarl import ISASClass


def build_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--algorithm", choices=("mappo", "masac"), default="mappo")
    parser.add_argument("--config", default="configs/isas_3x3.yaml")
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--buffer-device",
        default=None,
        help=(
            "Replay/on-policy buffer device. Defaults to --device. For large "
            "off-policy MASAC buffers on an 11 GB GPU, use 'cpu'."
        ),
    )
    parser.add_argument("--frames", type=int, default=3_000_000)
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument(
        "--loggers",
        nargs="+",
        choices=("tensorboard", "csv", "wandb", "mflow"),
        default=["tensorboard", "csv"],
    )
    parser.add_argument(
        "--log-dir",
        default="runs",
        help="Root folder in which BenchMARL creates the experiment directory.",
    )
    parser.add_argument(
        "--run-path-file",
        default=None,
        help=(
            "Optional text file to which the generated BenchMARL experiment "
            "directory is written. Useful for follow-up rendering scripts."
        ),
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=600_000,
        help="Collected frames between checkpoints. Use 0 to disable.",
    )
    parser.add_argument(
        "--checkpoint-at-end",
        action=BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--keep-checkpoints", type=int, default=3)
    parser.add_argument(
        "--evaluation",
        action=BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--evaluation-interval", type=int, default=300_000)
    parser.add_argument("--evaluation-episodes", type=int, default=8)
    parser.add_argument(
        "--render-evaluations",
        action=BooleanOptionalAction,
        default=False,
        help=(
            "Rendering every evaluation adds substantial overhead. Keep this "
            "disabled during training and render the final checkpoint separately."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    with Path(args.config).open("r", encoding="utf-8") as file:
        task_values = yaml.safe_load(file)
    task = ISASClass(name="SORTING", config=task_values)

    algorithm = (
        MappoConfig.get_from_yaml()
        if args.algorithm == "mappo"
        else MasacConfig.get_from_yaml()
    )
    model = MlpConfig.get_from_yaml()
    critic_model = MlpConfig.get_from_yaml()

    experiment_config = ExperimentConfig.get_from_yaml()
    experiment_config.sampling_device = args.device
    experiment_config.train_device = args.device
    experiment_config.buffer_device = args.buffer_device or args.device
    experiment_config.prefer_continuous_actions = False
    experiment_config.parallel_collection = False
    experiment_config.max_n_frames = args.frames
    experiment_config.max_n_iters = None

    experiment_config.evaluation = args.evaluation
    experiment_config.evaluation_interval = args.evaluation_interval
    experiment_config.evaluation_episodes = args.evaluation_episodes
    experiment_config.evaluation_deterministic_actions = True
    experiment_config.render = args.render_evaluations

    experiment_config.checkpoint_interval = args.checkpoint_interval
    experiment_config.checkpoint_at_end = args.checkpoint_at_end
    experiment_config.keep_checkpoints_num = args.keep_checkpoints

    log_dir = Path(args.log_dir).expanduser().resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    experiment_config.loggers = list(args.loggers)
    experiment_config.save_folder = str(log_dir)

    if args.algorithm == "mappo":
        experiment_config.on_policy_n_envs_per_worker = args.num_envs
        collected_frames = experiment_config.on_policy_collected_frames_per_batch
    else:
        experiment_config.off_policy_n_envs_per_worker = args.num_envs
        collected_frames = experiment_config.off_policy_collected_frames_per_batch

    for label, interval in (
        ("checkpoint interval", experiment_config.checkpoint_interval),
        (
            "evaluation interval",
            experiment_config.evaluation_interval
            if experiment_config.evaluation
            else 0,
        ),
    ):
        if interval and interval % collected_frames != 0:
            raise ValueError(
                f"{label} ({interval}) must be a multiple of the configured "
                f"collected frames per batch ({collected_frames})."
            )

    experiment = Experiment(
        task=task,
        algorithm_config=algorithm,
        model_config=model,
        critic_model_config=critic_model,
        seed=int(task_values.get("seed", 0)),
        config=experiment_config,
    )

    if args.run_path_file is not None:
        run_path_file = Path(args.run_path_file).expanduser().resolve()
        run_path_file.parent.mkdir(parents=True, exist_ok=True)
        run_path_file.write_text(str(experiment.folder_name.resolve()) + "\n")

    print(f"Algorithm: {args.algorithm}")
    print(f"Device: {args.device}")
    print(f"Buffer device: {experiment_config.buffer_device}")
    print(f"Vectorized environments: {args.num_envs}")
    print(f"Maximum collected frames: {args.frames}")
    print(f"Experiment directory: {experiment.folder_name.resolve()}")
    print(f"Loggers: {experiment_config.loggers}")
    print(
        "Note: BenchMARL frames count vectorized environment transitions, "
        "not physics substeps and not one frame per agent."
    )

    experiment.run()

    checkpoint_dir = experiment.folder_name / "checkpoints"
    checkpoints = sorted(
        checkpoint_dir.glob("checkpoint_*.pt"),
        key=lambda path: path.stat().st_mtime,
    )
    print(f"Finished with total_frames={experiment.total_frames}")
    if checkpoints:
        print(f"Final/latest checkpoint: {checkpoints[-1].resolve()}")
    else:
        print("No checkpoint was written.")


if __name__ == "__main__":
    main()

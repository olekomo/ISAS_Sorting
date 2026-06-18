"""Programmatic BenchMARL entry point; switch algorithms with one CLI flag."""

from argparse import ArgumentParser
from pathlib import Path

import torch
import yaml
from benchmarl.algorithms import MappoConfig, MasacConfig
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models import MlpConfig

from isas_benchmarl.benchmarl import ISASClass


parser = ArgumentParser()
parser.add_argument("--algorithm", choices=("mappo", "masac"), default="mappo")
parser.add_argument("--config", default="configs/isas_3x3.yaml")
parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
parser.add_argument("--frames", type=int, default=3_000_000)
parser.add_argument("--num-envs", type=int, default=256)
args = parser.parse_args()

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
experiment_config.buffer_device = args.device
experiment_config.prefer_continuous_actions = False
experiment_config.parallel_collection = False
experiment_config.max_n_frames = args.frames
experiment_config.render = False
experiment_config.evaluation = True
if args.algorithm == "mappo":
    experiment_config.on_policy_n_envs_per_worker = args.num_envs
else:
    experiment_config.off_policy_n_envs_per_worker = args.num_envs

experiment = Experiment(
    task=task,
    algorithm_config=algorithm,
    model_config=model,
    critic_model_config=critic_model,
    seed=int(task_values.get("seed", 0)),
    config=experiment_config,
)
experiment.run()

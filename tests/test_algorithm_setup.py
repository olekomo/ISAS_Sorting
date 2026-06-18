from benchmarl.algorithms import MappoConfig, MasacConfig
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models import MlpConfig

from isas_benchmarl.benchmarl import ISASClass


def _experiment_config() -> ExperimentConfig:
    config = ExperimentConfig.get_from_yaml()
    config.sampling_device = "cpu"
    config.train_device = "cpu"
    config.buffer_device = "cpu"
    config.prefer_continuous_actions = False
    config.parallel_collection = False
    config.evaluation = False
    config.render = False
    config.loggers = []
    config.create_json = False
    config.max_n_frames = 2
    config.on_policy_n_envs_per_worker = 1
    config.on_policy_collected_frames_per_batch = 2
    config.on_policy_minibatch_size = 2
    config.on_policy_n_minibatch_iters = 1
    config.off_policy_n_envs_per_worker = 1
    config.off_policy_collected_frames_per_batch = 2
    config.off_policy_train_batch_size = 2
    config.off_policy_n_optimizer_steps = 1
    config.off_policy_memory_size = 20
    return config


def test_mappo_and_masac_construct_from_same_task(config, tmp_path):
    model = MlpConfig.get_from_yaml()
    model.num_cells = [16, 16]

    critic = MlpConfig.get_from_yaml()
    critic.num_cells = [16, 16]

    for algorithm in (
        MappoConfig.get_from_yaml(),
        MasacConfig.get_from_yaml(),
    ):
        experiment_config = _experiment_config()

        # BenchMARL creates an experiment directory during construction.
        # Tests must not write into the repository root.
        experiment_config.save_folder = str(tmp_path)
        experiment_config.loggers = []
        experiment_config.create_json = False

        experiment = Experiment(
            task=ISASClass("SORTING", config),
            algorithm_config=algorithm,
            model_config=model,
            critic_model_config=critic,
            seed=config.seed,
            config=experiment_config,
        )

        experiment.close()

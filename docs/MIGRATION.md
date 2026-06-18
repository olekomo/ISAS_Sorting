# Migration into the existing project

The new package is intentionally isolated so the legacy simulator can remain
available as a behavioral reference during migration.

## 1. Copy the package

Copy `src/isas_benchmarl/` into the existing repository's `src/` tree and add
the dependencies from this repository's `pyproject.toml` to the BenchMARL
image.

The old configuration object can remain in place. `ISASConfig.from_source()`
accepts:

- a complete mapping with the existing field names;
- the existing typed `EnvConfig` instance;
- the existing `{ "config_dict": ... }` task wrapper;
- a `profile_name` mapping when the existing profile loader is importable.

## 2. Replace the BenchMARL class implementation

The old file may be reduced to a compatibility re-export:

```python
from isas_benchmarl.benchmarl import (
    ISASSortingClass,
    ISASTask,
)

__all__ = ["ISASSortingClass", "ISASTask"]
```

`ISASSortingClass` is an alias of the new native `ISASClass`. Existing registry
code can therefore keep the previous class name.

The essential change is that `num_envs` and `device` are passed into
`ISASTorchEnv` itself. Do not wrap it in `PettingZooWrapper`, `SerialEnv` or
`ParallelEnv` for the training path.

## 3. Task YAML

Use `conf/task/isas/sorting.yaml` from this repository or add the three new
values to the existing task mapping:

```yaml
max_fire_delay_steps: 8
max_decision_steps: 1000
y_distance_weight: 100.0
```

All previous simulator variables keep their names.

## 4. Optional PettingZoo import compatibility

Tools that still import the old PettingZoo class can use a thin re-export:

```python
from isas_benchmarl.compat.pettingzoo import (
    ISASSortingEnv,
    env,
    parallel_env,
)

__all__ = ["ISASSortingEnv", "env", "parallel_env"]
```

This adapter is B=1 and CPU-oriented. It must not be used by BenchMARL
training.

## 5. Algorithm selection

Keep one task/environment configuration and select only the BenchMARL
algorithm configuration. `examples/train_benchmarl.py` demonstrates MAPPO and
discrete MASAC using the same task and specs.

For both algorithms set:

```python
experiment_config.prefer_continuous_actions = False
```

## 6. Rendering

Training does not call Matplotlib. For evaluation or offline rendering, call
`env.snapshot(env_index)` and pass the snapshot to `MatplotlibRenderer` and
`FrameRecorder`. Only the selected environment is copied to CPU.

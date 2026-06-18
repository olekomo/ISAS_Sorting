# Explicit interface changes

## Preserved names and meanings

The following legacy configuration fields are accepted unchanged:

- grid and actor geometry;
- `T`, actor timing values and `frame_factor`;
- particle spawn, velocity, noise and birth-rate values;
- disturbance geometry;
- `obs_num_particles`, `num_particles`, `time_out`;
- reward coefficients;
- rendering-related values.

`ISASConfig.from_source()` accepts either a mapping or the existing typed
`EnvConfig` object.

## Required changes

### 1. Training environment class

Old training path:

```text
PettingZoo ParallelEnv -> PettingZooWrapper -> BenchMARL
```

New training path:

```text
ISASTorchEnv(EnvBase) -> BenchMARL
```

The optional `compat.ISASSortingEnv` preserves a B=1 PettingZoo surface for old
tools, but it is intentionally excluded from training.

### 2. Discrete action size and shape

Old action: binary per agent.

New action:

```text
Categorical(max_fire_delay_steps + 2)
shape [num_envs, num_actors]
```

The singleton dimension `[... ,1]` is omitted for categorical actions because
BenchMARL's discrete actor builds logits as `[agents, num_actions]`.

### 3. New configuration fields

```yaml
max_fire_delay_steps: 8
max_decision_steps: 1000
y_distance_weight: 100.0  # optional; previous hard-coded observation value
```

### 4. Pending schedule in observations

Delayed commands create hidden scheduler state. To keep the process Markov, the
normalized pending delay of every actor is appended to local observations and
the centralized state.

```text
old local observation: 6*K + 1 + 2 + 2*A
new local observation: 6*K + 1 + 2 + 3*A

old centralized state: 6*K + A
new centralized state: 6*K + 2*A
```

The existing six particle feature blocks and their order are retained.

### 5. Existence feature

The sixth particle feature now truthfully represents `particle_active`. The old
wrapper constructed this field from the number of allocated entries and could
mark resolved entries as valid.

### 6. Termination

- `terminated`: all configured particles have spawned and been resolved;
- `truncated`: `max_decision_steps` or the legacy inactivity `time_out` is
  reached first.

### 7. Spawn implementation

The same normal distributions and valid spawn region are used. Initial states
are pre-sampled at reset to avoid expensive recursive dynamic allocation during
steps. The random number sequence is therefore not bit-identical to the NumPy
implementation, although the configured distribution and constraints are
preserved.

### 8. Rendering

Rendering is no longer called inside each physics step. `snapshot()` performs an
explicit copy for one environment, after which renderer and recorder operate
entirely independently.

# ISAS BenchMARL Simulator

Native, batched PyTorch/TorchRL implementation of the current 2-D ISAS grid
sorting environment. The training path contains no PettingZoo wrapper and no
NumPy simulation. One `ISASTorchEnv` instance advances all environments in a
single tensor batch.

## Design goals

- native BenchMARL `TaskClass`;
- true environment batching on CPU or CUDA;
- MAPPO and discrete MASAC use the same task and observation/action specs;
- a configurable number of physics steps between policy decisions;
- delayed discrete actuator commands;
- rendering, images and videos isolated from the training hot path;
- compatibility loader for the existing `EnvConfig` field names;
- optional B=1 PettingZoo adapter for old tooling.

## Repository structure

```text
isas_benchmarl_sim/
├── pyproject.toml
├── configs/
│   ├── isas_3x3.yaml
│   └── isas_full.yaml
├── conf/task/isas/sorting.yaml
├── docs/
│   ├── ARCHITECTURE.md
│   ├── INTERFACE_CHANGES.md
│   ├── MIGRATION.md
│   └── VALIDATION.md
├── examples/
│   ├── benchmark.py
│   ├── record_random_episode.py
│   ├── smoke_env.py
│   └── train_benchmarl.py
├── src/isas_benchmarl/
│   ├── config.py                 # legacy-compatible typed config
│   ├── enums.py
│   ├── state.py                  # fixed-shape tensor state
│   ├── actors.py                 # actor cycle + delayed scheduler
│   ├── particles.py              # births, CV dynamics, wall reflection
│   ├── contacts.py               # hit/disturbance contact engine
│   ├── rewards.py                # shared reward + episode metrics
│   ├── observations.py           # local Top-K obs + central state
│   ├── dynamics.py               # complete framework-neutral simulator
│   ├── env.py                    # native TorchRL EnvBase
│   ├── benchmarl/
│   │   └── task.py               # ISASClass / ISASTask
│   ├── rendering/
│   │   ├── snapshot.py
│   │   ├── renderer.py
│   │   └── recorder.py
│   └── compat/
│       └── pettingzoo.py         # optional, not used for training
└── tests/
    ├── test_actor_scheduler.py
    ├── test_algorithm_setup.py
    ├── test_benchmarl_task.py
    ├── test_contacts_rewards.py
    ├── test_env.py
    ├── test_pettingzoo_compat.py
    ├── test_rendering.py
    └── test_time_scale.py
```

## Time model

`T` is the duration of one physics interval. `frame_factor` is the number of
physics intervals executed after one model action.

```text
policy decision period = T * frame_factor
```

Example:

```yaml
T: 0.1
frame_factor: 5
```

The simulator advances every 0.1 seconds, while the model receives a new
observation and provides actions every 0.5 seconds.

The order inside every physics interval remains compatible with the legacy
sorter:

1. move already active particles;
2. advance/activate actors;
3. create new particles;
4. evaluate contacts and disturbance state;
5. resolve hits, disturbances and particles reaching `array_end`.

## Delayed discrete action

Set:

```yaml
max_fire_delay_steps: 8
```

The categorical action has `max_fire_delay_steps + 2` values:

| Action | Meaning |
|---:|---|
| `0` | do not create an activation plan |
| `1` | start actor activation in the current physics interval |
| `2` | start activation after one complete physics interval |
| `n` | start activation after `n - 1` complete physics intervals |

“Start activation” is not identical to physical impact. The actor still passes
through `t_activate` and `t_up`; its effective HIT interval starts afterward.

An actor holds at most one pending plan. While a plan is pending or the actor is
not READY, its action mask permits only action `0`. Plans are therefore never
silently overwritten.

## Native tensor layout

```text
B = num_envs
A = num_actors
P = num_particles
K = obs_num_particles

particle_state          [B, P, 4]   # x, vx, y, vy
particle_active         [B, P]
particle_resolved       [B, P]
potential_disturbance   [B, A, P]
actor_phase             [B, A]
pending_delay           [B, A]
action                  [B, A]
action_mask             [B, A, num_actions]
observation             [B, A, observation_dim]
state                   [B, state_dim]
reward                  [B, A, 1]
```

Particle slots never change shape. Birth and removal are represented through
masks, so no append, deletion, Python set or ID-association matrix is required.

## Installation

```bash
python -m pip install -e ".[dev]"
pytest
```

## Validation status

The packaged implementation passes 13 tests, TorchRL spec validation, B=1
PettingZoo compatibility, independent PNG/MP4 generation, and tiny BenchMARL
training runs with both MAPPO and discrete MASAC. See
`docs/VALIDATION.md` for the exact tested versions and the CPU-only hardware
limitation.

## TorchRL smoke test

```bash
python examples/smoke_env.py
```

## BenchMARL training

MAPPO:

```bash
python examples/train_benchmarl.py \
  --algorithm mappo \
  --config configs/isas_3x3.yaml \
  --device cuda:0 \
  --num-envs 512
```

Discrete MASAC:

```bash
python examples/train_benchmarl.py \
  --algorithm masac \
  --config configs/isas_3x3.yaml \
  --device cuda:0 \
  --num-envs 512
```

The environment selection, reward, action mask and simulation stay identical;
only the BenchMARL algorithm config changes.

## Images and videos outside training

```bash
python examples/record_random_episode.py
```

Rendering is performed from an immutable CPU snapshot of one selected batch
member. No frame is generated and no GPU-to-CPU copy occurs unless `snapshot()`
or `render()` is called explicitly.

## Performance benchmark

```bash
python examples/benchmark.py --device cuda:0 --num-envs 2048 --steps 1000
```

Measure model/training throughput separately from pure environment throughput.
For CUDA measurements, synchronization is used only around the complete timed
block.

## Scope

This repository fully implements the simulation used by the current BenchMARL
wrapper: 2-D area sorter, CV particle model, two or more configured class
process labels, actor timing, wall reflection, stateful disturbance areas,
shared reward, local observations, centralized state and rendering.

The unused generic legacy Line/CA/eval-string process hierarchy was deliberately
not copied into the GPU training core. Adding CA later should be done as a new
fixed-dimension tensor dynamics module, not by restoring dynamic Python process
objects.

## GPU Docker and TensorBoard

The repository includes a dependency-only GPU development image. Source files
are bind-mounted, so ordinary code and configuration changes do not require an
image rebuild.

- Dockerfile: `docker/Dockerfile.gpu`
- Complete server and TensorBoard guide: `docs/DOCKER_SERVER.md`

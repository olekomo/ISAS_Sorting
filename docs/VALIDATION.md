# Validation record

Final validation was executed on 2026-06-18 with:

- Python 3.13;
- PyTorch 2.10.0 CPU build;
- TorchRL 0.11.1;
- TensorDict 0.11.0;
- BenchMARL 1.5.2.

## Passed checks

- editable package installation;
- Python byte-code compilation of source, tests and examples;
- TorchRL `check_env_specs` on a native batched environment;
- 13 unit/integration tests;
- partial resets in a batch;
- delayed actions inside one `frame_factor` window;
- delayed actions spanning multiple policy decisions;
- reward/contact behavior tests;
- PNG rendering and MP4 recording;
- B=1 PettingZoo compatibility step;
- BenchMARL Experiment construction for MAPPO and discrete MASAC;
- actual tiny optimization/collection runs for MAPPO and MASAC.

The tiny algorithm runs intentionally used a horizon too short to complete an
episode, so BenchMARL emitted its expected `mean return = nan` warning. Both
collectors and optimizer steps completed successfully.

## Hardware limitation

The execution environment available for this build contained a CPU-only
PyTorch wheel. CUDA execution and throughput were therefore not measured here.
The simulation hot path uses device-local PyTorch tensors and generators and
contains no NumPy or host transfer, but a CUDA smoke test and benchmark remain
mandatory on the target GPU machine:

```bash
python examples/benchmark.py --device cuda:0 --num-envs 2048 --steps 1000
```

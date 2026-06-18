# GPU development container and TensorBoard

This setup is intended for a Linux server with an NVIDIA GPU. The Docker image
contains CUDA-enabled PyTorch, TorchRL, BenchMARL, TensorBoard and rendering
dependencies. The repository itself is mounted into the container at runtime.
Source-code and YAML changes therefore do not require a new image build.

## File locations in this repository

```text
isas_benchmarl_sim/
├── docker/
│   └── Dockerfile.gpu          # build this image once
├── docs/
│   └── DOCKER_SERVER.md        # this guide
├── examples/
│   └── train_benchmarl.py      # logger and log-directory configuration
└── runs/                       # created on the server; TensorBoard event files
```

Place `Dockerfile.gpu` exactly at `docker/Dockerfile.gpu`, relative to the
repository root. Run all following commands from the repository root.

## 1. Server prerequisites

The host needs:

- Docker Engine;
- an NVIDIA driver that supports the CUDA runtime used by the image;
- NVIDIA Container Toolkit configured for Docker.

Check the host GPU first:

```bash
nvidia-smi
```

Check Docker GPU access:

```bash
docker run --rm --gpus all \
  pytorch/pytorch:2.10.0-cuda12.6-cudnn9-runtime \
  nvidia-smi
```

Do not continue with training until this command can see the GPU.

## 2. Build the dependency image once

From the repository root:

```bash
docker build --pull \
  -f docker/Dockerfile.gpu \
  -t isas-benchmarl:gpu \
  .
```

The Dockerfile deliberately does not copy the repository. Rebuild the image
only when Python/system dependencies or the Dockerfile change. Changes under
`src/`, `examples/`, `configs/`, `conf/` or `tests/` do not require a rebuild.

## 3. Start one persistent development container

Create the host log directory first:

```bash
mkdir -p runs
```

Start the container:

```bash
docker run -d \
  --name isas-dev \
  --gpus all \
  --shm-size=8g \
  -p 127.0.0.1:6006:6006 \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -e PYTHONPATH=/workspace/src \
  --mount "type=bind,source=$(pwd),target=/workspace" \
  --workdir /workspace \
  isas-benchmarl:gpu \
  sleep infinity
```

Important details:

- `--gpus all` exposes the NVIDIA GPUs to PyTorch.
- `--shm-size=8g` avoids a very small default shared-memory segment.
- `--mount ... /workspace` exposes the live repository to the container.
- `--user ...` keeps files under `runs/` owned by your server user rather than
  root.
- `-p 127.0.0.1:6006:6006` exposes TensorBoard only on the server's loopback
  interface. Access it locally through an SSH tunnel rather than opening it to
  the internet.

If only one physical GPU should be visible, replace `--gpus all` with, for
example:

```bash
--gpus '"device=1"'
```

Inside that container, the selected GPU is normally addressed as `cuda:0`.

## 4. Verify the environment

```bash
docker exec -it isas-dev python -c '
import torch
print("torch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
assert torch.cuda.is_available()
'
```

Then run the environment smoke test:

```bash
docker exec -it isas-dev python examples/smoke_env.py
```

## 5. TensorBoard logger configuration

The programmatic BenchMARL entry point is:

```text
examples/train_benchmarl.py
```

The relevant configuration is:

```python
experiment_config.loggers = list(args.loggers)
experiment_config.save_folder = str(log_dir)
```

The script now accepts:

```text
--loggers tensorboard
--log-dir /workspace/runs
```

You may enable more than one logger:

```text
--loggers tensorboard csv
```

BenchMARL creates an experiment-specific subdirectory below `save_folder`.
TensorBoard scans `runs/` recursively, so it is not necessary to know the final
experiment directory in advance.

## 6. Start training

MAPPO:

```bash
docker exec -it isas-dev \
  python examples/train_benchmarl.py \
    --algorithm mappo \
    --config configs/isas_3x3.yaml \
    --device cuda:0 \
    --num-envs 512 \
    --frames 3000000 \
    --loggers tensorboard \
    --log-dir /workspace/runs
```

Discrete MASAC:

```bash
docker exec -it isas-dev \
  python examples/train_benchmarl.py \
    --algorithm masac \
    --config configs/isas_3x3.yaml \
    --device cuda:0 \
    --num-envs 512 \
    --frames 3000000 \
    --loggers tensorboard \
    --log-dir /workspace/runs
```

Use a server-side `tmux` or `screen` session for long interactive runs. Source
changes are immediately visible to the container, but an already running Python
training process does not hot-reload modules; restart only the training process,
not the image or container.

## 7. Start TensorBoard inside the same container

Run this in a second server shell:

```bash
docker exec -it isas-dev \
  tensorboard \
    --logdir /workspace/runs \
    --host 0.0.0.0 \
    --port 6006
```

For a detached TensorBoard process:

```bash
docker exec -d isas-dev bash -lc \
  'tensorboard --logdir /workspace/runs --host 0.0.0.0 --port 6006 \
   > /workspace/runs/tensorboard.log 2>&1'
```

Inspect its output on the server with:

```bash
tail -f runs/tensorboard.log
```

TensorBoard must listen on `0.0.0.0` inside the container so Docker's port
forwarding can reach it. The Docker port is still bound only to
`127.0.0.1` on the server.

## 8. Open TensorBoard in your local browser

On your local machine, not on the server, create an SSH tunnel:

```bash
ssh -N -L 6006:127.0.0.1:6006 YOUR_USER@YOUR_SERVER
```

Keep that terminal open and visit:

```text
http://localhost:6006
```

If local port 6006 is occupied:

```bash
ssh -N -L 16006:127.0.0.1:6006 YOUR_USER@YOUR_SERVER
```

Then visit `http://localhost:16006`.

## 9. Container lifecycle

Open a shell:

```bash
docker exec -it isas-dev bash
```

Stop and restart without losing the container configuration:

```bash
docker stop isas-dev
docker start isas-dev
```

Remove the container when it is no longer needed:

```bash
docker rm -f isas-dev
```

The source code and `runs/` remain on the server because they are bind-mounted
host files, not container data.

## 10. When an image rebuild is required

No rebuild:

- Python source edits;
- YAML configuration edits;
- test changes;
- model/environment logic changes;
- README or documentation changes.

Rebuild:

- dependency/version changes in `docker/Dockerfile.gpu` or `pyproject.toml`;
- new `apt` packages;
- a different PyTorch/CUDA base image.

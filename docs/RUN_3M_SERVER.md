# 3-Million-Frame MAPPO run on the GPU server

This run uses one physical GPU and at most 20 CPU cores. The current project and
BenchMARL entry point are single-device: exposing a second GPU does not
automatically split one experiment across both GPUs.

Recommended physical GPU: GPU 1, because GPU 0 is already occupied in the
provided `nvidia-smi` output. Inside a container that exposes only physical GPU
1, this device is named `cuda:0`.

## Start the development container

Run from the repository root on the server:

```bash
mkdir -p runs artifacts

docker rm -f isas-dev 2>/dev/null || true

docker run -d   --name isas-dev   --gpus '"device=1"'   --cpus=20   --shm-size=8g   --ulimit memlock=-1   --ulimit stack=67108864   -p 127.0.0.1:6006:6006   --user "$(id -u):$(id -g)"   -e HOME=/tmp   -e PYTHONPATH=/workspace/src   -e OMP_NUM_THREADS=20   -e MKL_NUM_THREADS=20   -e OPENBLAS_NUM_THREADS=20   -e NUMEXPR_NUM_THREADS=20   --mount "type=bind,source=$(pwd),target=/workspace"   --workdir /workspace   isas-benchmarl:gpu   sleep infinity
```

## Start TensorBoard

```bash
docker exec -d isas-dev bash -lc '
tensorboard   --logdir /workspace/runs   --host 0.0.0.0   --port 6006   > /workspace/runs/tensorboard.log 2>&1
'
```

On the local computer:

```bash
ssh -N -L 6006:127.0.0.1:6006 prak_ss_3@i81-gpu-server
```

Open `http://localhost:6006`.

## Start the complete run and final video

```bash
docker exec -d isas-dev bash -lc '
DEVICE=cuda:0 NUM_ENVS=512 FRAMES=3000000 bash scripts/run_3m_mappo.sh
'
```

Follow the run:

```bash
tail -f runs/mappo_3m_console.log
```

Find the experiment:

```bash
cat runs/.mappo_3m_run_path
```

Expected results are below that directory:

```text
checkpoints/checkpoint_3000000.pt
final_render/learned_episode.mp4
final_render/learned_final.png
```

Notes:

- `3,000,000` means collected BenchMARL environment frames, not physics
  substeps in one episode.
- This launcher uses one GPU. A second visible GPU would remain unused.
- If 512 vectorized environments exceed GPU memory, rerun with
  `NUM_ENVS=256`.

#!/usr/bin/env bash
set -Eeuo pipefail

DEVICE="${DEVICE:-cuda:0}"
NUM_ENVS="${NUM_ENVS:-512}"
FRAMES="${FRAMES:-3000000}"
CONFIG="${CONFIG:-configs/isas_3x3.yaml}"
RUN_ROOT="${RUN_ROOT:-/workspace/runs}"
RUN_PATH_FILE="${RUN_PATH_FILE:-${RUN_ROOT}/.mappo_3m_run_path}"
CONSOLE_LOG="${CONSOLE_LOG:-${RUN_ROOT}/mappo_3m_console.log}"
FPS="${FPS:-10}"

mkdir -p "${RUN_ROOT}"
rm -f "${RUN_PATH_FILE}"

echo "Starting MAPPO training"
echo "device=${DEVICE}"
echo "num_envs=${NUM_ENVS}"
echo "frames=${FRAMES}"
echo "logs=${RUN_ROOT}"

python examples/train_benchmarl.py   --algorithm mappo   --config "${CONFIG}"   --device "${DEVICE}"   --buffer-device "${DEVICE}"   --num-envs "${NUM_ENVS}"   --frames "${FRAMES}"   --loggers tensorboard csv   --log-dir "${RUN_ROOT}"   --run-path-file "${RUN_PATH_FILE}"   --checkpoint-interval 600000   --checkpoint-at-end   --keep-checkpoints 3   --evaluation   --evaluation-interval 300000   --evaluation-episodes 8   --no-render-evaluations   2>&1 | tee "${CONSOLE_LOG}"

if [[ ! -s "${RUN_PATH_FILE}" ]]; then
  echo "Training finished but the experiment path file is missing: ${RUN_PATH_FILE}" >&2
  exit 1
fi

RUN_DIR="$(cat "${RUN_PATH_FILE}")"
CHECKPOINT="${RUN_DIR}/checkpoints/checkpoint_${FRAMES}.pt"

if [[ ! -f "${CHECKPOINT}" ]]; then
  CHECKPOINT="$(
    find "${RUN_DIR}/checkpoints" -maxdepth 1 -type f -name 'checkpoint_*.pt'       -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-
  )"
fi

if [[ -z "${CHECKPOINT}" || ! -f "${CHECKPOINT}" ]]; then
  echo "No checkpoint found below ${RUN_DIR}/checkpoints" >&2
  exit 1
fi

OUTPUT_DIR="${RUN_DIR}/final_render"
echo "Rendering deterministic learned episode from ${CHECKPOINT}"

python examples/record_checkpoint_episode.py   --checkpoint "${CHECKPOINT}"   --device "${DEVICE}"   --output-dir "${OUTPUT_DIR}"   --video-name learned_episode.mp4   --image-name learned_final.png   --fps "${FPS}"   2>&1 | tee -a "${CONSOLE_LOG}"

echo
echo "Completed."
echo "Experiment: ${RUN_DIR}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Video: ${OUTPUT_DIR}/learned_episode.mp4"
echo "Image: ${OUTPUT_DIR}/learned_final.png"

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONFIG="${CONFIG:-configs/elfcaps_audiocaps2.yml}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-checkpoints/release/checkpoint}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/elfcaps_train_reproduce}"

mkdir -p "$OUTPUT_DIR"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
python -u src/train_audio_trajectory_manifold.py \
  --config "$CONFIG" \
  --checkpoint "$INIT_CHECKPOINT" \
  --output_dir "$OUTPUT_DIR" \
  --epochs "${EPOCHS:-1}" \
  --batch_size "${BATCH_SIZE:-4}" \
  --max_train_examples "${MAX_TRAIN_EXAMPLES:-2048}" \
  --lr "${LR:-5e-8}" \
  --warmup_steps "${WARMUP_STEPS:-16}" \
  --num_sampling_steps "${NUM_SAMPLING_STEPS:-8}" \
  --cfg_scale "${CFG_SCALE:-3}" \
  --self_cond_cfg_scale "${SELF_COND_CFG_SCALE:-1}" \
  --mse_weight "${MSE_WEIGHT:-0.02}" \
  --cos_weight "${COS_WEIGHT:-0.15}" \
  --rms_weight "${RMS_WEIGHT:-0.3}" \
  --mean_weight "${MEAN_WEIGHT:-0.02}" \
  --freeze_audio_adapter \
  --seed "${SEED:-60}" \
  --log_freq "${LOG_FREQ:-16}"

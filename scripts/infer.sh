#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

EVAL_CSV="${1:-${AUDIOCAPS_TEST_CSV:-data/audiocaps1_test.csv}}"
NUM_SAMPLES="${2:-10}"
OUTPUT="${3:-outputs/elfcaps_infer_examples.jsonl}"
CONFIG="${CONFIG:-configs/elfcaps_audiocaps2.yml}"
CHECKPOINT="${CHECKPOINT:-checkpoints/release/checkpoint}"

mkdir -p "$(dirname "$OUTPUT")"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
python -u src/eval_audio_checkpoint.py \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --eval_data_path "$EVAL_CSV" \
  --num_samples "$NUM_SAMPLES" \
  --batch_size "${BATCH_SIZE:-8}" \
  --num_sampling_steps "${NUM_SAMPLING_STEPS:-16}" \
  --cfg_scale "${CFG_SCALE:-3}" \
  --self_cond_cfg_scale "${SELF_COND_CFG_SCALE:-1}" \
  --max_decode_tokens "${MAX_DECODE_TOKENS:-32}" \
  --min_decode_tokens "${MIN_DECODE_TOKENS:-6}" \
  --clean_artifacts \
  --use_raw_params \
  --output "$OUTPUT"

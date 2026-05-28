#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONFIG="${CONFIG:-configs/elfcaps_audiocaps2.yml}"
CHECKPOINT="${CHECKPOINT:-checkpoints/release/checkpoint}"
EVAL_CSV="${AUDIOCAPS_TEST_CSV:-data/audiocaps1_test.csv}"
REF_CSV="${AUDIOCAPS_REFS_CSV:-data/audiocaps1_test_refs.csv}"
PRED="${PRED:-outputs/evals/elfcaps_audiocaps1test957_steps16_cfg3_mindec6_clean.jsonl}"
REFS="${REFS:-outputs/evals/elfcaps_audiocaps1test957_refs5.jsonl}"
METRICS="${METRICS:-outputs/evals/elfcaps_audiocaps1test957_refs5.metrics.json}"

mkdir -p "$(dirname "$PRED")" "$(dirname "$REFS")" "$(dirname "$METRICS")"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
python -u src/eval_audio_checkpoint.py \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --eval_data_path "$EVAL_CSV" \
  --num_samples "${NUM_SAMPLES:-957}" \
  --batch_size "${BATCH_SIZE:-8}" \
  --num_sampling_steps "${NUM_SAMPLING_STEPS:-16}" \
  --cfg_scale "${CFG_SCALE:-3}" \
  --self_cond_cfg_scale "${SELF_COND_CFG_SCALE:-1}" \
  --max_decode_tokens "${MAX_DECODE_TOKENS:-32}" \
  --min_decode_tokens "${MIN_DECODE_TOKENS:-6}" \
  --clean_artifacts \
  --use_raw_params \
  --output "$PRED"

python src/add_audiocaps_references.py \
  --predictions "$PRED" \
  --references_csv "$REF_CSV" \
  --output "$REFS"

python src/evaluate_caption_table_metrics.py \
  --predictions "$REFS" \
  --coco_metrics \
  --output "$METRICS"

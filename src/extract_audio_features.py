#!/usr/bin/env python
"""Precompute PE-A-Frame audio features for AudioCaps-style CSV splits."""

import argparse
import csv
import math
import os
import sys

import numpy as np
import torch
from tqdm import tqdm
from transformers import PeAudioFrameLevelModel, PeAudioProcessor


def parse_args():
    parser = argparse.ArgumentParser(description="Extract frame-level PE audio embeddings.")
    parser.add_argument("--data_dir", default="data/audiocaps2")
    parser.add_argument("--output_dir", default="data/audiocaps_pe_features")
    parser.add_argument("--model_name", default="facebook/pe-a-frame-small")
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=None, help="Optional max rows for smoke tests.")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=100)
    return parser.parse_args()


def iter_rows(csv_path):
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        yield from csv.DictReader(f)


def batched(items, batch_size):
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def main():
    args = parse_args()
    if args.num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard_index must satisfy 0 <= shard_index < num_shards")

    os.makedirs(args.output_dir, exist_ok=True)
    processor = PeAudioProcessor.from_pretrained(args.model_name)
    model = PeAudioFrameLevelModel.from_pretrained(args.model_name).to(args.device)
    model.eval()

    audio_dir = os.path.join(args.data_dir, "audiocaps_raw_audio")
    splits = [split.strip() for split in args.splits.split(",") if split.strip()]
    rows = []
    for split in splits:
        rows.extend(list(iter_rows(os.path.join(args.data_dir, f"{split}.csv"))))
    if args.limit is not None:
        rows = rows[:args.limit]
    rows = rows[args.shard_index::args.num_shards]

    with torch.inference_mode():
        total_batches = math.ceil(len(rows) / args.batch_size) if rows else 0
        saved, skipped_existing, skipped_missing, skipped_empty_caption = 0, 0, 0, 0
        use_tqdm = sys.stderr.isatty()
        row_batches = batched(rows, args.batch_size)
        if use_tqdm:
            row_batches = tqdm(
                row_batches,
                total=total_batches,
                desc=f"extract audio features shard {args.shard_index + 1}/{args.num_shards}",
            )
        for batch_idx, row_batch in enumerate(row_batches, start=1):
            batch_items = []
            for row in row_batch:
                if not row.get("caption"):
                    skipped_empty_caption += 1
                    continue
                stem = f"{row['youtube_id']}_{row['start_time']}"
                audio_path = os.path.join(audio_dir, f"{stem}.wav")
                output_path = os.path.join(args.output_dir, f"{stem}.npy")
                if os.path.exists(output_path):
                    skipped_existing += 1
                    continue
                if not os.path.exists(audio_path):
                    skipped_missing += 1
                    continue
                batch_items.append((audio_path, output_path))
            if not batch_items:
                if use_tqdm:
                    row_batches.set_postfix(
                        saved=saved, existing=skipped_existing, missing=skipped_missing,
                    )
                elif args.log_every > 0 and batch_idx % args.log_every == 0:
                    print(
                        f"progress batch={batch_idx}/{total_batches} saved={saved} "
                        f"existing={skipped_existing} missing_audio={skipped_missing}",
                        flush=True,
                    )
                continue

            audio_paths = [item[0] for item in batch_items]
            output_paths = [item[1] for item in batch_items]
            inputs = processor(audio=audio_paths, return_tensors="pt", padding=True)
            inputs = {k: v.to(args.device) if hasattr(v, "to") else v for k, v in inputs.items()}
            audio_outputs = model.audio_encoder(
                input_values=inputs["input_values"],
                padding_mask=inputs.get("padding_mask"),
            )
            audio_embeds = model.audio_head(audio_outputs.last_hidden_state)
            output_mask = getattr(audio_outputs, "output_mask", None)
            if output_mask is not None:
                lengths = output_mask.to(torch.int64).sum(dim=1).detach().cpu().numpy()
            else:
                lengths = [audio_embeds.shape[1]] * audio_embeds.shape[0]
            audio_embeds = audio_embeds.detach().cpu().numpy().astype(np.float32)
            for features, length, output_path in zip(audio_embeds, lengths, output_paths):
                np.save(output_path, features[:int(length)])
                saved += 1
            if use_tqdm:
                row_batches.set_postfix(saved=saved, existing=skipped_existing, missing=skipped_missing)
            elif args.log_every > 0 and batch_idx % args.log_every == 0:
                print(
                    f"progress batch={batch_idx}/{total_batches} saved={saved} "
                    f"existing={skipped_existing} missing_audio={skipped_missing}",
                    flush=True,
                )

    print(
        "summary: "
        f"rows={len(rows)} saved={saved} existing={skipped_existing} "
        f"missing_audio={skipped_missing} empty_caption={skipped_empty_caption}"
    )


if __name__ == "__main__":
    main()

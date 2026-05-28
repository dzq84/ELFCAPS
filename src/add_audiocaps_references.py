#!/usr/bin/env python
"""Attach multi-reference AudioCaps captions to prediction JSONL files."""

import argparse
import csv
import json
import os
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(description="Add AudioCaps references to a prediction JSONL.")
    parser.add_argument("--predictions", required=True, help="Input JSONL with an input wav path per row.")
    parser.add_argument("--references_csv", required=True, help="AudioCaps CSV with youtube_id,start_time,caption.")
    parser.add_argument("--output", required=True, help="Output JSONL with references attached.")
    return parser.parse_args()


def _load_references(path):
    refs = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            caption = str(row.get("caption", "")).strip()
            if not caption:
                continue
            key = (str(row["youtube_id"]), str(row["start_time"]))
            if caption not in refs[key]:
                refs[key].append(caption)
    return dict(refs)


def _key_from_wav(path):
    stem = os.path.splitext(os.path.basename(str(path)))[0]
    youtube_id, start_time = stem.rsplit("_", 1)
    return youtube_id, start_time


def main():
    args = parse_args()
    refs = _load_references(args.references_csv)
    total = 0
    missing = 0
    ref_counts = defaultdict(int)
    with open(args.predictions, "r", encoding="utf-8") as src, open(
        args.output, "w", encoding="utf-8"
    ) as dst:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            key = _key_from_wav(row["input"])
            row_refs = refs.get(key)
            if not row_refs:
                missing += 1
                row_refs = [str(row.get("target", ""))]
            row["references"] = row_refs
            ref_counts[len(row_refs)] += 1
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")
            total += 1
    print(
        json.dumps(
            {
                "predictions": args.predictions,
                "references_csv": args.references_csv,
                "output": args.output,
                "rows": total,
                "missing_reference_rows": missing,
                "reference_count_histogram": dict(sorted(ref_counts.items())),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

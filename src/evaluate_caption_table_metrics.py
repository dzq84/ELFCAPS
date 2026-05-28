#!/usr/bin/env python
"""Evaluate caption JSONL with paper-style captioning metrics.

This script reports BLEU-1..4, ROUGE-L, and METEOR on a 0-100 scale so the
output is easier to compare with audio-captioning tables. If pycocoevalcap is
installed, it also reports CIDEr, SPICE, and SPIDEr.
"""

import argparse
import json
import math
import re
from collections import Counter

from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer


TOKEN_RE = re.compile(r"[a-z0-9']+")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate caption predictions with table-style metrics.")
    parser.add_argument("--predictions", required=True, help="JSONL with prediction and target/reference fields.")
    parser.add_argument("--output", default=None, help="Optional path to write JSON metrics.")
    parser.add_argument(
        "--coco_metrics",
        action="store_true",
        help="Also compute pycocoevalcap CIDEr/SPICE/SPIDEr when dependencies are installed.",
    )
    return parser.parse_args()


def _tokenize(text):
    return TOKEN_RE.findall(str(text).lower())


def _references_from_row(row):
    for key in ("references", "targets", "captions"):
        refs = row.get(key)
        if isinstance(refs, list) and refs:
            return [str(ref) for ref in refs]
    target = row.get("target", row.get("reference", ""))
    return [str(target)]


def _load_rows(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _closest_reference_length(pred_len, ref_lens):
    return min(ref_lens, key=lambda ref_len: (abs(ref_len - pred_len), ref_len))


def _modified_precision(pred_tokens, ref_token_lists, n):
    if len(pred_tokens) < n:
        return 0, 0
    pred_counts = Counter(tuple(pred_tokens[i : i + n]) for i in range(len(pred_tokens) - n + 1))
    max_ref_counts = Counter()
    for ref_tokens in ref_token_lists:
        ref_counts = Counter(tuple(ref_tokens[i : i + n]) for i in range(max(len(ref_tokens) - n + 1, 0)))
        for gram, count in ref_counts.items():
            if count > max_ref_counts[gram]:
                max_ref_counts[gram] = count
    clipped = sum(min(count, max_ref_counts[gram]) for gram, count in pred_counts.items())
    total = sum(pred_counts.values())
    return clipped, total


def _corpus_bleu_n(predictions, references, max_n):
    clipped = [0] * max_n
    totals = [0] * max_n
    pred_len_total = 0
    ref_len_total = 0

    for pred, refs in zip(predictions, references):
        pred_tokens = _tokenize(pred)
        ref_token_lists = [_tokenize(ref) for ref in refs]
        pred_len = len(pred_tokens)
        ref_lens = [len(ref_tokens) for ref_tokens in ref_token_lists] or [0]
        pred_len_total += pred_len
        ref_len_total += _closest_reference_length(pred_len, ref_lens)
        for n in range(1, max_n + 1):
            num, den = _modified_precision(pred_tokens, ref_token_lists, n)
            clipped[n - 1] += num
            totals[n - 1] += den

    if pred_len_total == 0:
        return 0.0
    bp = 1.0 if pred_len_total > ref_len_total else math.exp(1.0 - ref_len_total / max(pred_len_total, 1))
    precisions = []
    for num, den in zip(clipped, totals):
        if den == 0:
            precisions.append(0.0)
        elif num == 0:
            precisions.append(1.0 / (2.0 * den))
        else:
            precisions.append(num / den)
    if any(p <= 0.0 for p in precisions):
        return 0.0
    return 100.0 * bp * math.exp(sum(math.log(p) for p in precisions) / max_n)


def _rouge_l(predictions, references):
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    vals = []
    for pred, refs in zip(predictions, references):
        vals.append(max(scorer.score(ref, pred)["rougeL"].fmeasure for ref in refs))
    return 100.0 * sum(vals) / max(len(vals), 1)


def _meteor(predictions, references):
    vals = []
    for pred, refs in zip(predictions, references):
        ref_tokens = [_tokenize(ref) for ref in refs]
        pred_tokens = _tokenize(pred)
        vals.append(meteor_score(ref_tokens, pred_tokens))
    return 100.0 * sum(vals) / max(len(vals), 1)


def _coco_metric_inputs(predictions, references):
    gts = {}
    res = {}
    for idx, (pred, refs) in enumerate(zip(predictions, references)):
        key = str(idx)
        gts[key] = [str(ref) for ref in refs]
        res[key] = [str(pred)]
    return gts, res


def _try_coco_metrics(predictions, references):
    try:
        from pycocoevalcap.cider.cider import Cider
        from pycocoevalcap.spice.spice import Spice
    except Exception as exc:
        return {
            "cider_spice_available": False,
            "coco_metrics_error": f"{type(exc).__name__}: {exc}",
        }

    gts, res = _coco_metric_inputs(predictions, references)
    out = {"cider_spice_available": False}
    try:
        cider, _ = Cider().compute_score(gts, res)
        out["cider"] = 100.0 * float(cider)
    except Exception as exc:
        out["cider_error"] = f"{type(exc).__name__}: {exc}"

    try:
        spice, _ = Spice().compute_score(gts, res)
        out["spice"] = 100.0 * float(spice)
    except Exception as exc:
        out["spice_error"] = f"{type(exc).__name__}: {exc}"

    if out.get("cider") is not None and out.get("spice") is not None:
        out["spider"] = 0.5 * (out["cider"] + out["spice"])
        out["cider_spice_available"] = True
    return out


def evaluate(rows, coco_metrics=False):
    predictions = [str(row.get("prediction", "")) for row in rows]
    references = [_references_from_row(row) for row in rows]
    metrics = {
        "num_examples": len(rows),
        "num_references_min": min((len(refs) for refs in references), default=0),
        "num_references_max": max((len(refs) for refs in references), default=0),
        "bleu1": _corpus_bleu_n(predictions, references, 1),
        "bleu2": _corpus_bleu_n(predictions, references, 2),
        "bleu3": _corpus_bleu_n(predictions, references, 3),
        "bleu4": _corpus_bleu_n(predictions, references, 4),
        "rougeL": _rouge_l(predictions, references),
        "meteor": None,
        "meteor_available": False,
        "cider": None,
        "spice": None,
        "spider": None,
        "cider_spice_available": False,
        "scale": "0-100",
        "notes": [
            "BLEU is corpus BLEU-N with closest-reference brevity penalty and light tokenization.",
            "ROUGE-L and METEOR use the best score over references when multiple references are present.",
            "CIDEr, SPICE, and SPIDEr are computed with pycocoevalcap only when --coco_metrics is set.",
        ],
    }
    try:
        metrics["meteor"] = _meteor(predictions, references)
        metrics["meteor_available"] = True
    except LookupError as exc:
        metrics["meteor_error"] = str(exc).splitlines()[0]
    if coco_metrics:
        metrics.update(_try_coco_metrics(predictions, references))
    return metrics


def main():
    args = parse_args()
    rows = _load_rows(args.predictions)
    metrics = evaluate(rows, coco_metrics=args.coco_metrics)
    text = json.dumps(metrics, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()

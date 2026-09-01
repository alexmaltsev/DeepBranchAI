#!/usr/bin/env python3
"""Compute full-volume DeepBranchAI metrics from a CSV of prediction/reference pairs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from deepbranchai.image_io import load_volume
from deepbranchai.metrics import compute_binary_volume_metrics

REQUIRED_COLUMNS = ("case", "prediction", "reference")


def _resolve(path: str, base: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else base / candidate


def compute(pairs_path: Path, output_path: Path) -> None:
    with pairs_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing = [column for column in REQUIRED_COLUMNS if column not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Pair table is missing columns: {', '.join(missing)}")
        pairs = list(reader)

    rows = []
    for pair in pairs:
        prediction_path = _resolve(pair["prediction"], pairs_path.parent)
        reference_path = _resolve(pair["reference"], pairs_path.parent)
        metrics = compute_binary_volume_metrics(load_volume(prediction_path), load_volume(reference_path))
        metadata = {key: value for key, value in pair.items() if key not in {"prediction", "reference"}}
        rows.append({**metadata, **metrics})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("Pair table contains no rows")
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} volume-wise rows to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pairs", type=Path, help="CSV with case, prediction, and reference columns")
    parser.add_argument("output", type=Path, help="Output per-volume metrics CSV")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    compute(args.pairs, args.output)

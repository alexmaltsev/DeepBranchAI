#!/usr/bin/env python3
"""Recompute the paired external-transfer tests reported in manuscript Table 6."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "results/external_transfer/external_transfer_paired_per_volume.csv"
EXPECTED_RESULTS = REPO_ROOT / "results/external_transfer/wilcoxon_holm_results.csv"
DATASET_ORDER = ("3D-IRCADb venous CT", "Plant CT roots", "AeroPath airway CT")
METRICS = ("dice", "cldice", "abs_cc_error")


def holm_adjust(raw_p_values: list[float]) -> list[float]:
    order = np.argsort(raw_p_values)
    adjusted = np.empty(len(raw_p_values), dtype=float)
    running_max = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (len(raw_p_values) - rank) * raw_p_values[index])
        running_max = max(running_max, candidate)
        adjusted[index] = running_max
    return adjusted.tolist()


def calculate(paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset in DATASET_ORDER:
        subset = paired.loc[paired["dataset"] == dataset]
        if subset.empty:
            raise ValueError(f"No paired rows found for {dataset}")
        for metric in METRICS:
            deepbranchai = subset[f"deepbranchai_{metric}"].to_numpy(dtype=float)
            scratch = subset[f"scratch_{metric}"].to_numpy(dtype=float)
            zero_differences = int(np.count_nonzero(deepbranchai == scratch))
            # Make the accepted-paper calculation independent of SciPy's
            # version-specific ``method="auto"`` selection.
            method = "approx" if zero_differences else "exact"
            result = wilcoxon(
                deepbranchai,
                scratch,
                alternative="two-sided",
                zero_method="wilcox",
                method=method,
            )
            rows.append(
                {
                    "dataset": dataset,
                    "paired_volumes": len(subset),
                    "metric": metric,
                    "wilcoxon_statistic": float(result.statistic),
                    "raw_p": float(result.pvalue),
                    "zero_differences": zero_differences,
                }
            )
    adjusted = holm_adjust([row["raw_p"] for row in rows])
    for row, adjusted_p in zip(rows, adjusted):
        row["holm_adjusted_p"] = adjusted_p
    return pd.DataFrame(rows)


def verify(actual: pd.DataFrame, expected_path: Path) -> None:
    expected = pd.read_csv(expected_path)
    keys = ["dataset", "paired_volumes", "metric"]
    actual = actual.sort_values(keys).reset_index(drop=True)
    expected = expected.sort_values(keys).reset_index(drop=True)
    if not actual[keys].equals(expected[keys]):
        raise ValueError("Computed result rows do not match the tracked result rows")
    for column in ("wilcoxon_statistic", "raw_p", "holm_adjusted_p"):
        if not np.allclose(actual[column], expected[column], rtol=0, atol=1e-12):
            raise ValueError(f"Computed {column} values do not match {expected_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--expected", type=Path, default=EXPECTED_RESULTS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = calculate(pd.read_csv(args.input))
    verify(results, args.expected)
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        results.to_csv(args.output_dir / "wilcoxon_holm_results.csv", index=False)
    print(results.to_string(index=False))
    print("\nTracked Table 6 statistics reproduced successfully.")


if __name__ == "__main__":
    main()

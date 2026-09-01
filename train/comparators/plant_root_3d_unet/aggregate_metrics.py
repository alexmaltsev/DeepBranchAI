from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


SELECTED_METRICS = [
    "dice",
    "precision",
    "recall",
    "specificity",
    "hd95",
    "cldice",
    "pred_components",
    "pred_fragmentation_index",
    "pred_skeleton_components",
    "pred_skeleton_largest_component_fraction",
    "target_skeleton_components",
    "target_skeleton_largest_component_fraction",
]


def fnum(value: str | int | float | None) -> float:
    try:
        return float(value)
    except Exception:
        return math.nan


def read_fold_rows(experiment_root: Path, folds: list[int]) -> tuple[list[dict[str, str]], list[str]]:
    all_rows: list[dict[str, str]] = []
    fieldnames = ["fold"]
    seen = {"fold"}
    for fold in folds:
        path = experiment_root / f"fold_{fold}" / "validation_metrics.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing fold metrics CSV: {path}")
        with path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            out = {"fold": str(fold), **row}
            all_rows.append(out)
            for key in out:
                if key not in seen:
                    fieldnames.append(key)
                    seen.add(key)
    return all_rows, fieldnames


def write_rows(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict], numeric_fields: list[str]) -> dict[str, float | int]:
    out: dict[str, float | int] = {"case_count": len(rows)}
    for key in numeric_fields:
        values_all = [fnum(row.get(key)) for row in rows]
        finite = [value for value in values_all if math.isfinite(value)]
        out[f"{key}_finite_count"] = len(finite)
        out[f"{key}_nonfinite_count"] = len(values_all) - len(finite)
        if finite:
            mean = sum(finite) / len(finite)
            variance = sum((value - mean) ** 2 for value in finite) / len(finite)
            out[f"{key}_mean"] = mean
            out[f"{key}_std"] = math.sqrt(variance)
            out[f"{key}_min"] = min(finite)
            out[f"{key}_max"] = max(finite)
        else:
            out[f"{key}_mean"] = math.nan
            out[f"{key}_std"] = math.nan
            out[f"{key}_min"] = math.nan
            out[f"{key}_max"] = math.nan
    return out


def run(args: argparse.Namespace) -> None:
    rows, fieldnames = read_fold_rows(args.experiment_root, args.folds)
    all_csv = args.experiment_root / "continuity_metrics_all_folds.csv"
    write_rows(all_csv, rows, fieldnames)

    numeric_fields = [key for key in fieldnames if key not in {"fold", "case_id", "source_id"}]
    fold_rows = []
    for fold in args.folds:
        fold_case_rows = [row for row in rows if row["fold"] == str(fold)]
        fold_rows.append({"fold": fold, **summarize(fold_case_rows, numeric_fields)})

    summary_fields: list[str] = []
    seen_summary = set()
    for row in fold_rows:
        for key in row:
            if key not in seen_summary:
                summary_fields.append(key)
                seen_summary.add(key)
    fold_summary_csv = args.experiment_root / "continuity_metrics_fold_summary.csv"
    write_rows(fold_summary_csv, fold_rows, summary_fields)

    overall = summarize(rows, numeric_fields)
    compact = {
        "case_count": len(rows),
        "all_cases_csv": str(all_csv),
        "fold_summary_csv": str(fold_summary_csv),
        "selected_overall": {
            key: {
                "mean": overall.get(f"{key}_mean", math.nan),
                "std": overall.get(f"{key}_std", math.nan),
                "finite_count": overall.get(f"{key}_finite_count", 0),
                "nonfinite_count": overall.get(f"{key}_nonfinite_count", 0),
            }
            for key in SELECTED_METRICS
        },
        "folds": [
            {
                "fold": row["fold"],
                "case_count": row["case_count"],
                **{f"{key}_mean": row.get(f"{key}_mean", math.nan) for key in SELECTED_METRICS},
                **{f"{key}_nonfinite_count": row.get(f"{key}_nonfinite_count", 0) for key in SELECTED_METRICS},
            }
            for row in fold_rows
        ],
    }
    summary_json = args.experiment_root / "continuity_metrics_summary.json"
    summary_json.write_text(json.dumps(compact, indent=2, allow_nan=True), encoding="utf-8")

    print(f"Wrote {all_csv}")
    print(f"Wrote {fold_summary_csv}")
    print(f"Wrote {summary_json}")
    for key in SELECTED_METRICS:
        item = compact["selected_overall"][key]
        print(
            f"{key}: mean={item['mean']:.6g} std={item['std']:.6g} "
            f"finite={item['finite_count']} nonfinite={item['nonfinite_count']}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate 3D U-Net full-volume validation metrics across folds.")
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.experiment_root = args.experiment_root.resolve()
    run(args)


if __name__ == "__main__":
    main()

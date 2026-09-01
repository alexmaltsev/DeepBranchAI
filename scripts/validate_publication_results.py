#!/usr/bin/env python3
"""Validate internal consistency of the accepted-paper result files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = ROOT / "results/manuscript"
SOURCE = ROOT / "results/source_domain"
EXTERNAL = ROOT / "results/external_transfer"
METRICS = ("precision", "sensitivity", "specificity", "dice", "cldice", "abs_cc_error", "avd_percent", "kappa")


def assert_display_value(actual: float, reported: float, decimals: int, label: str) -> None:
    tolerance = 0.5 * 10 ** (-decimals) + 1e-12
    if not np.isclose(actual, reported, rtol=0, atol=tolerance):
        raise ValueError(f"{label} does not match its source value: {actual} vs {reported}")


def source_folds(filename: str) -> pd.DataFrame:
    table = pd.read_csv(SOURCE / filename, dtype={"fold": str})
    return table.loc[table["fold"] != "all"].copy()


def validate_table2() -> None:
    table = pd.read_csv(MANUSCRIPT / "table2_mitochondrial_cross_validation.csv")
    folds = table.loc[table["row"].str.startswith("fold_")]
    mean = table.loc[table["row"] == "mean"].iloc[0]
    sample_sd = table.loc[table["row"] == "sample_sd"].iloc[0]
    if len(folds) != 5 or int(folds["n_volumes"].sum()) != 20:
        raise ValueError("Table 2 must contain five folds and 20 held-out volumes")
    for metric in METRICS:
        if not np.isclose(folds[metric].mean(), mean[metric], atol=1e-12):
            raise ValueError(f"Table 2 {metric} mean is inconsistent with its fold rows")
        if not np.isclose(folds[metric].std(ddof=1), sample_sd[metric], atol=1e-12):
            raise ValueError(f"Table 2 {metric} SD is inconsistent with its fold rows")

    source = source_folds("deepbranchai_ce_dice_per_fold_and_all.csv")
    if not np.allclose(folds[list(METRICS)], source[list(METRICS)], rtol=0, atol=1e-12):
        raise ValueError("Table 2 fold rows do not match the tracked source-domain summary")


def validate_table3() -> None:
    table = pd.read_csv(MANUSCRIPT / "table3_architecture_and_loss_comparison.csv").set_index("model")
    sources = {
        "DeepBranchAI (3D nnU-Net; CE + Dice)": "deepbranchai_ce_dice_per_fold_and_all.csv",
        "DeepBranchAI (3D nnU-Net; CE + Dice + clDice)": "deepbranchai_ce_dice_cldice_per_fold_and_all.csv",
    }
    for model, filename in sources.items():
        folds = source_folds(filename)
        for metric in METRICS:
            decimals = 0 if metric == "abs_cc_error" else 3
            assert_display_value(
                folds[metric].mean(), table.loc[model, f"{metric}_mean"], decimals, f"Table 3 {model} {metric} mean"
            )
            assert_display_value(
                folds[metric].std(ddof=1),
                table.loc[model, f"{metric}_sd"],
                decimals,
                f"Table 3 {model} {metric} SD",
            )


def validate_table4() -> None:
    table = pd.read_csv(MANUSCRIPT / "table4_z_context_ablation.csv").set_index("z_depth")
    sources = {
        32: source_folds("deepbranchai_z32_per_fold.csv"),
        128: source_folds("deepbranchai_ce_dice_per_fold_and_all.csv"),
    }
    for z_depth, folds in sources.items():
        for metric in METRICS:
            decimals = 0 if metric == "abs_cc_error" else 3
            assert_display_value(
                folds[metric].mean(),
                table.loc[z_depth, f"{metric}_mean"],
                decimals,
                f"Table 4 Z{z_depth} {metric} mean",
            )
            assert_display_value(
                folds[metric].std(ddof=1),
                table.loc[z_depth, f"{metric}_sd"],
                decimals,
                f"Table 4 Z{z_depth} {metric} SD",
            )


def validate_table5() -> None:
    table = pd.read_csv(MANUSCRIPT / "table5_external_transfer.csv")
    reported = pd.read_csv(MANUSCRIPT / "table5_relative_difference.csv").set_index("dataset")
    for dataset, rows in table.groupby("dataset", sort=False):
        deepbranchai = rows.loc[rows["model"] == "DeepBranchAI-pretrained fine-tune"].iloc[0]
        comparators = rows.loc[rows["model"] != "DeepBranchAI-pretrained fine-tune"]
        calculated = {}
        for metric in ("dice", "cldice"):
            baseline = comparators[f"{metric}_mean"].max()
            calculated[f"{metric}_percent"] = (deepbranchai[f"{metric}_mean"] - baseline) / baseline * 100
        baseline = comparators["abs_cc_error_mean"].min()
        calculated["abs_cc_error_percent"] = (baseline - deepbranchai["abs_cc_error_mean"]) / baseline * 100
        for metric, value in calculated.items():
            if not np.isclose(round(value, 2), reported.loc[dataset, metric], atol=0.005):
                raise ValueError(f"Table 5 {dataset} {metric} percentage is inconsistent")

    paired = pd.read_csv(EXTERNAL / "external_transfer_paired_per_volume.csv")
    dataset_names = {
        "3D-IRCADb venous CT": "3D-IRCADb venous system",
        "Plant CT roots": "Plant root CT",
        "AeroPath airway CT": "AeroPath airway CT",
    }
    model_prefixes = {
        "Scratch nnU-Net": "scratch",
        "DeepBranchAI-pretrained fine-tune": "deepbranchai",
    }
    indexed = table.set_index(["dataset", "model"])
    for source_name, table_name in dataset_names.items():
        subset = paired.loc[paired["dataset"] == source_name]
        if len(subset) != int(indexed.loc[(table_name, "Scratch nnU-Net"), "n"]):
            raise ValueError(f"Table 5 sample size does not match paired rows for {table_name}")
        for model, prefix in model_prefixes.items():
            fold_means = subset.groupby("fold")[
                [f"{prefix}_{metric}" for metric in ("dice", "cldice", "abs_cc_error")]
            ].mean()
            for metric in ("dice", "cldice", "abs_cc_error"):
                decimals = 2 if metric == "abs_cc_error" else 3
                values = fold_means[f"{prefix}_{metric}"]
                assert_display_value(
                    values.mean(),
                    indexed.loc[(table_name, model), f"{metric}_mean"],
                    decimals,
                    f"Table 5 {table_name} {model} {metric} mean",
                )
                assert_display_value(
                    values.std(ddof=1),
                    indexed.loc[(table_name, model), f"{metric}_sd"],
                    decimals,
                    f"Table 5 {table_name} {model} {metric} SD",
                )


def validate_table6() -> None:
    manuscript = pd.read_csv(MANUSCRIPT / "table6_paired_tests.csv").sort_values("dataset").reset_index(drop=True)
    raw = pd.read_csv(ROOT / "results/external_transfer/wilcoxon_holm_results.csv")
    pivot = raw.pivot(index=["dataset", "paired_volumes"], columns="metric", values="holm_adjusted_p").reset_index()
    pivot = (
        pivot.rename(
            columns={
                "dice": "dice_adjusted_p",
                "cldice": "cldice_adjusted_p",
                "abs_cc_error": "abs_cc_error_adjusted_p",
            }
        )
        .sort_values("dataset")
        .reset_index(drop=True)
    )
    if not manuscript[["dataset", "paired_volumes"]].equals(pivot[["dataset", "paired_volumes"]]):
        raise ValueError("Table 6 dataset rows do not match the paired statistical results")
    for column in ("dice_adjusted_p", "cldice_adjusted_p", "abs_cc_error_adjusted_p"):
        if not np.allclose(manuscript[column], pivot[column], rtol=0, atol=1e-12):
            raise ValueError(f"Table 6 {column} does not match the paired statistical results")


def validate_portability() -> None:
    forbidden = ("C:\\Users\\", "D:\\", "E:\\", "F:\\")
    for path in (ROOT / "results").rglob("*.csv"):
        text = path.read_text(encoding="utf-8")
        if any(value in text for value in forbidden):
            raise ValueError(f"Machine-specific path found in {path.relative_to(ROOT)}")


def main() -> None:
    validate_table2()
    validate_table3()
    validate_table4()
    validate_table5()
    validate_table6()
    validate_portability()
    print("Publication result tables and provenance files are internally consistent.")


if __name__ == "__main__":
    main()

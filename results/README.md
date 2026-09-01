# Accepted-Paper Results

This directory contains the machine-readable values reported in the accepted manuscript, *DeepBranchAI: A Transferable 3D Segmentation Model for Branching Networks*, plus the available per-volume evidence used to calculate them.

## Reporting Conventions

- Source-domain and external metrics were computed on assembled full 3D validation volumes, not isolated training patches or 2D slices.
- Fold summaries are unweighted means across folds 0-4. Dispersion is the sample standard deviation across those five fold means.
- clDice is calculated from skeletonized prediction and reference masks.
- Absolute connected-component error is `abs(predicted components - reference components)` using 26-connectivity.
- Table 5 percentages are calculated from the displayed mean values. Positive values favor DeepBranchAI; negative values indicate that a comparator had the better displayed mean.
- Table 6 uses paired, two-sided Wilcoxon signed-rank tests on held-out per-volume metrics with one Holm correction across all nine dataset-metric tests.
- The two Plant CT cases with nonempty references and no predicted centerline are retained as failures with clDice set to zero.

## Files

### `manuscript/`

| File | Content |
|---|---|
| `table1_datasets.csv` | Source and external dataset roles |
| `table2_mitochondrial_cross_validation.csv` | Five source-domain folds, overall mean, and sample SD |
| `table3_architecture_and_loss_comparison.csv` | 2D/3D architecture and clDice-loss comparison |
| `table4_z_context_ablation.csv` | Five-fold Z32, Z64, and Z128 summary |
| `table5_external_transfer.csv` | External model means and sample SDs |
| `table5_relative_difference.csv` | Displayed percentage differences |
| `table6_paired_tests.csv` | Exact Holm-adjusted paired-test values |

### `source_domain/`

- `deepbranchai_ce_dice_per_volume.csv`: 20 full-volume predictions from the released CE + Dice model.
- `deepbranchai_ce_dice_cldice_per_volume.csv`: 20 full-volume predictions from the CE + Dice + clDice comparison.
- `deepbranchai_z32_per_volume.csv`: 20 full-volume predictions from the Z32 context comparison.
- Matching fold summary files preserve the unrounded calculation inputs.

The accepted Table 4 is the canonical five-fold summary for Z32, Z64, and Z128. The public evidence bundle includes full per-volume tables for Z32 and Z128 and the accepted five-fold Z64 summary.

### `external_transfer/`

- `external_transfer_paired_per_volume.csv`: paired scratch and DeepBranchAI out-of-fold values for 12 3D-IRCADb, 29 Plant CT, and 27 AeroPath volumes.
- `wilcoxon_holm_results.csv`: exact statistics and adjusted p-values used for Table 6.

### `qualitative/figure5/`

The final Figure 5 source masks and case-level Dice/clDice values. The rendered figure is stored at [`docs/assets/figure5_external_transfer.png`](../docs/assets/figure5_external_transfer.png).

## Verification

From the repository root:

```bash
python scripts/compute_external_transfer_statistics.py
python scripts/validate_publication_results.py
```

The result CSVs contain no machine-specific filesystem paths. Checkpoints and large source datasets are linked rather than committed.

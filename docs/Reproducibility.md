# Reproducibility Guide

## Result Trace

| Accepted result | Machine-readable source | Calculation |
|---|---|---|
| Table 2 | `results/source_domain/deepbranchai_ce_dice_per_volume.csv` | Per-volume metrics grouped into five held-out folds |
| Table 3 clDice-loss row | `results/source_domain/deepbranchai_ce_dice_cldice_per_volume.csv` | Same 20 held-out volumes and full-volume metric code |
| Table 4 Z32/Z128 | `results/source_domain/deepbranchai_z32_per_volume.csv` and CE + Dice source table | Per-volume fold aggregation |
| Table 4 Z64 | `results/manuscript/table4_z_context_ablation.csv` | Accepted five-fold summary |
| Table 5 | `results/manuscript/table5_external_transfer.csv` | Unweighted mean and sample SD across five fold means |
| Table 6 | `results/external_transfer/external_transfer_paired_per_volume.csv` | Paired Wilcoxon tests and Holm correction |
| Figure 5 | `results/qualitative/figure5/` | Submitted render, source masks, and case metrics |

## Verify Tracked Results

Create an environment with the repository dependencies, then run:

```bash
python scripts/compute_external_transfer_statistics.py
python scripts/validate_publication_results.py
python scripts/audit_repository_images.py
python -m pytest -q
```

`compute_external_transfer_statistics.py` recomputes all nine paired tests from 68 held-out volume pairs and verifies the exact tracked statistics. `validate_publication_results.py` checks the available source links for Tables 2-5, Table 5 percentage arithmetic, Table 6 linkage, and the absence of machine-specific paths. `audit_repository_images.py` rejects TIFF files, unapproved raster assets, and embedded notebook images.

## Recompute Metrics From Masks

Create a CSV with these columns:

```csv
case,prediction,reference
case_001,path/to/prediction.nii.gz,path/to/reference.nii.gz
```

Then run:

```bash
python scripts/compute_volume_metrics.py pairs.csv per_volume_metrics.csv
```

The command computes precision, sensitivity, specificity, accuracy, Dice, clDice, 26-connected component counts, absolute connected-component error, AVD, and Kappa on each assembled 3D volume.

## Checkpoints And Data

The five source checkpoints and nnU-Net configuration are on [Zenodo](https://zenodo.org/records/19363534). External datasets remain at their original sources and are not redistributed here. The qualitative Figure 5 masks are included because they are compact and directly support the submitted figure.

Large raw volumes, predictions, optimizer states, and third-party checkpoints are intentionally excluded from Git. The CSV provenance files use anonymous case identifiers and preserve fold assignments without retaining source filenames or local filesystem paths.

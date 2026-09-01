# Accepted Paper Results

The exact machine-readable values are stored under [`results/manuscript/`](../results/manuscript/). Values below use the accepted manuscript's display precision.

## Source-Domain Validation

| Metric | Five-fold mean +/- sample SD |
|---|---:|
| Dice | 0.942 +/- 0.020 |
| Precision | 0.962 +/- 0.016 |
| Sensitivity | 0.925 +/- 0.050 |
| Specificity | 0.996 +/- 0.002 |
| clDice | 0.940 +/- 0.021 |
| Absolute connected-component error | 214 +/- 122 |
| AVD (%) | 6.032 +/- 4.565 |
| Kappa | 0.935 +/- 0.022 |

These values were calculated volume-wise across five subject-grouped folds with four full 128-slice held-out volumes per fold.

## Architecture And Loss Comparison

| Model | Dice | clDice | Abs. CC error |
|---|---:|---:|---:|
| 2D U-Net | 0.789 +/- 0.051 | 0.735 +/- 0.095 | 7565 +/- 3790 |
| 3D U-Net | 0.907 +/- 0.047 | 0.879 +/- 0.068 | 1267 +/- 648 |
| 2D nnU-Net | 0.890 +/- 0.046 | 0.879 +/- 0.041 | 1373 +/- 577 |
| DeepBranchAI, CE + Dice | 0.942 +/- 0.020 | 0.940 +/- 0.021 | 214 +/- 122 |
| DeepBranchAI, CE + Dice + clDice | 0.941 +/- 0.019 | 0.941 +/- 0.028 | 183 +/- 108 |

The additional clDice loss had a marginal effect across most metrics. Its clearest difference was lower mean absolute connected-component error.

## Z-Context Comparison

| Z depth | Dice | clDice | Abs. CC error | AVD (%) |
|---:|---:|---:|---:|---:|
| 32 | 0.933 +/- 0.035 | 0.931 +/- 0.033 | 230 +/- 226 | 6.664 +/- 5.076 |
| 64 | 0.936 +/- 0.025 | 0.936 +/- 0.029 | 203 +/- 96 | 6.718 +/- 4.375 |
| 128 | 0.942 +/- 0.020 | 0.940 +/- 0.021 | 214 +/- 122 | 6.032 +/- 4.565 |

The released 352 x 352 x 128 configuration had the strongest overall balance across overlap, continuity, and agreement metrics. Absolute connected-component error was similar for Z64 and Z128.

## External Transfer

| Dataset | Model | Dice | clDice | Abs. CC error |
|---|---|---:|---:|---:|
| 3D-IRCADb, n=12 | VesselFM fine-tune | 0.464 +/- 0.056 | 0.360 +/- 0.098 | 111.47 +/- 45.91 |
|  | Scratch nnU-Net | 0.638 +/- 0.093 | 0.563 +/- 0.100 | 25.60 +/- 3.24 |
|  | DeepBranchAI-pretrained | 0.679 +/- 0.080 | 0.629 +/- 0.100 | 27.17 +/- 5.06 |
| Plant CT roots, n=29 | 3D U-Net root baseline | 0.459 +/- 0.066 | 0.403 +/- 0.079 | 97.95 +/- 27.80 |
|  | Scratch nnU-Net | 0.557 +/- 0.023 | 0.568 +/- 0.060 | 25.47 +/- 17.05 |
|  | DeepBranchAI-pretrained | 0.611 +/- 0.053 | 0.599 +/- 0.062 | 19.11 +/- 13.07 |
| AeroPath, n=27 | MurineAirwaySegmentation fine-tune | 0.831 +/- 0.043 | 0.768 +/- 0.031 | 76.13 +/- 18.72 |
|  | Scratch nnU-Net | 0.879 +/- 0.031 | 0.824 +/- 0.019 | 56.12 +/- 14.23 |
|  | DeepBranchAI-pretrained | 0.880 +/- 0.033 | 0.840 +/- 0.014 | 49.57 +/- 15.77 |

DeepBranchAI-pretrained fine-tuning had higher mean clDice than scratch nnU-Net in all three datasets. Paired tests supported clDice differences for 3D-IRCADb and AeroPath, Dice for 3D-IRCADb and Plant CT, and absolute connected-component error for AeroPath after Holm correction. See Table 6 for exact adjusted p-values.

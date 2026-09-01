# Figure 5 Source Data

Figure 5 compares held-out 3D-IRCADb venous-system segmentations in matched 3D views.

- Panel A: successful case `ircad_15_009`, fold 4.
- Panel B: challenging case `ircad_01_001`, fold 1.

Each case directory contains the reference segmentation and predictions from scratch nnU-Net, VesselFM fine-tuning, and DeepBranchAI-pretrained fine-tuning as compressed NIfTI masks. `figure5_metrics.csv` records the Dice and clDice values printed above each prediction.

The underlying cases are from the [3D-IRCADb dataset](https://www.ircad.fr/research-and-development/data-sets/liver-segmentation-3d-ircadb-01/). The source dataset's terms continue to apply to its reference data.

The rendered figure is [`docs/assets/figure5_external_transfer.png`](../../../docs/assets/figure5_external_transfer.png).

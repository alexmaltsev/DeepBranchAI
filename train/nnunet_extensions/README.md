# nnU-Net Training Extensions

This directory contains the custom trainer code used for the accepted comparison experiments.

- `compound_cldice_loss.py`: combined cross-entropy, Dice, and clDice objective.
- `nnUNetTrainer_CE_DC_CLDC.py`: nnU-Net trainer using the combined objective.
- `nnUNetTrainer_CE_DC_CLDC_Xepochs.py`: configurable-epoch variant.
- `nnUNetTrainer_ZDepthAblation.py`: trainer used to constrain Z context for the Z32 and Z64 comparisons.

These files target nnU-Net v2.3.1. Install them into the corresponding nnU-Net trainer module path or make them importable in the active environment before launching training. The released DeepBranchAI checkpoint itself uses the standard nnU-Net CE + Dice objective.

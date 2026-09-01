# Plant Root 3D U-Net Comparator

This is the custom PyTorch 3D U-Net used for the Plant CT comparator in the accepted paper. It is not a MONAI model.

## Model

- Four encoder levels with base channels `16, 32, 64, 128`.
- Two 3D convolutions per block.
- Max-pooling downsampling, transposed-convolution upsampling, and concatenated skip connections.
- Instance normalization and leaky ReLU.
- Approximately 1.40 million trainable parameters with the reported configuration.

## Reported Training Protocol

- Random initialization and full weight updates.
- 100 epochs, batch size 1.
- BCE-with-logits plus soft Dice loss.
- AdamW with learning rate `0.001` and weight decay `1e-5`.
- Constant learning rate.
- Foreground-biased patch sampling and no conventional flip, rotation, noise, or intensity augmentation.
- Checkpoint selected by highest patch-validation Dice, with the latest checkpoint winning ties.

The accepted summary values are stored in [`results/manuscript/table5_external_transfer.csv`](../../../results/manuscript/table5_external_transfer.csv). Dataset folds and local data paths must be supplied by the user.

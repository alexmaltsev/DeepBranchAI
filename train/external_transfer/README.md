# Matched External Transfer Runner

`run_nnunet_transfer.py` runs a prepared target nnU-Net dataset with either random initialization or a released DeepBranchAI checkpoint. Use the same `splits_final.json`, plans, trainer, epoch budget, and folds for both commands.

```bash
python train/external_transfer/run_nnunet_transfer.py 3210 --mode scratch
python train/external_transfer/run_nnunet_transfer.py 3210 --mode deepbranchai --source-fold 0
```

The target dataset must already exist under the configured `nnUNet_raw` and `nnUNet_preprocessed` roots. The DeepBranchAI command downloads fold 0 by default, matching the accepted external-transfer experiments. Use `--pretrained-checkpoint` to provide an already installed checkpoint.

Dataset-specific conversion and label definitions remain the responsibility of the target dataset. See [`docs/External_Transfer.md`](../../docs/External_Transfer.md) for the accepted protocols and comparator details.

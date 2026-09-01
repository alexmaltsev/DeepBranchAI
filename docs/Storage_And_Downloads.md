# Storage And Downloads

DeepBranchAI keeps code in the Git repository and large datasets, checkpoints, predictions, and nnU-Net intermediates in a configurable storage root.

## Storage Layout

`deepbranchai.paths.setup_environment(...)` creates:

```text
<storage>/
|-- data/
|-- weights/
|-- tmp/
|-- nnUNet_raw/
|-- nnUNet_preprocessed/
`-- nnUNet_results/
```

Without `storage_dir`, the repository root is used. For real training, use a separate high-capacity location:

```python
from deepbranchai.paths import setup_environment

paths = setup_environment(storage_dir="/data/DeepBranchAI")
```

On Windows, a raw string avoids backslash escaping:

```python
paths = setup_environment(storage_dir=r"F:\DeepBranchAI")
```

## Environment Variables

The same roots can be configured with:

- `DEEPBRANCHAI_STORAGE_DIR`
- `DEEPBRANCHAI_DATA_DIR`
- `DEEPBRANCHAI_WEIGHTS_DIR`
- `DEEPBRANCHAI_TMP_DIR`
- `DEEPBRANCHAI_NNUNET_RAW`
- `DEEPBRANCHAI_NNUNET_PREPROCESSED`
- `DEEPBRANCHAI_NNUNET_RESULTS`

`setup_environment` also sets the standard `nnUNet_raw`, `nnUNet_preprocessed`, and `nnUNet_results` variables for the current process.

## Download A Released Checkpoint

```python
from deepbranchai.downloads import download_and_install_pretrained_weights
from deepbranchai.paths import setup_environment

paths = setup_environment(storage_dir="/data/DeepBranchAI")
checkpoint = download_and_install_pretrained_weights(paths, fold=0)
```

Valid folds are 0-4. The helper downloads the selected checkpoint, the mitochondrial `nnUNetPlans.json`, and `dataset.json` from [Zenodo](https://zenodo.org/records/19363534), then installs them into nnU-Net's expected layout:

```text
<nnUNet_results>/Dataset4005_Mitochondria/
  nnUNetTrainer_100epochs__nnUNetPlans__3d_fullres/
    fold_0/
      checkpoint_best.pth
```

Fold 0 is the default and matches the accepted paper's external-transfer initialization. Repeated calls reuse files that are already present.

## VESSEL12 Assets

The Zenodo record also contains a VESSEL12 checkpoint, demo archive, and training archive. The VESSEL12 notebooks download these through the corresponding helpers in `deepbranchai.downloads`. They provide lung-vasculature demonstration and fine-tuning materials alongside the external-transfer resources.

## Git Exclusions

The repository ignores all local storage roots and generated predictions. Do not force-add raw datasets, downloaded checkpoints, nnU-Net caches, or third-party model files. Accepted summary tables and compact result provenance are tracked under `results/`.

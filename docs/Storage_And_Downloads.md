# Storage And Downloads

This guide explains how storage is resolved and what the downloader/stager installs.

## Storage Model

The code directory stays in the repository. Large assets can live elsewhere.

`deepbranchai.paths.setup_environment(...)` creates and returns these roots:

```text
base
storage
data
weights
tmp
nnUNet_raw
nnUNet_preprocessed
nnUNet_results
```

## Default Behavior

If you do not set `storage_dir`, then storage defaults to the repository root.

That means this layout:

```text
Repo/
├─ data/
├─ weights/
├─ tmp/
├─ nnUNet_raw/
├─ nnUNet_preprocessed/
└─ nnUNet_results/
```

## External Storage

If you set:

```python
STORAGE_DIR = Path(r"F:\DeepBranchAI")
```

then the asset layout becomes:

```text
F:\DeepBranchAI/
├─ data/
├─ weights/
├─ tmp/
├─ nnUNet_raw/
├─ nnUNet_preprocessed/
└─ nnUNet_results/
```

The repository still holds the code and notebooks. Only large assets move.

## Environment Variables

You can configure storage in code or with environment variables.

Supported variables:

- `DEEPBRANCHAI_STORAGE_DIR`
- `DEEPBRANCHAI_DATA_DIR`
- `DEEPBRANCHAI_WEIGHTS_DIR`
- `DEEPBRANCHAI_TMP_DIR`
- `DEEPBRANCHAI_NNUNET_RAW`
- `DEEPBRANCHAI_NNUNET_PREPROCESSED`
- `DEEPBRANCHAI_NNUNET_RESULTS`

`storage_dir` is the simplest option when the roots should stay grouped together.

## How Relative Paths Resolve

Inside `FinetuneConfig`:

- `raw_dir`, `ground_truth_dir`, `predict_dir`, and `output_dir` resolve under `data`
- `pretrained_weights` resolves under `weights` by default
- `pretrained_weights` resolves under `nnUNet_results` if it begins with `nnUNet_results/...`

Examples:

```python
raw_dir="custom_finetune/raw"
```

becomes:

```text
<data directory>/custom_finetune/raw
```

```python
pretrained_weights="checkpoint_best.pth"
```

becomes:

```text
<weights directory>/checkpoint_best.pth
```

## Downloader And Stager

The finetune notebook exposes:

```python
DOWNLOAD_AND_STAGE_VESSEL12 = True
```

That calls `download_and_stage_vessel12_example(config)`.

## What Gets Downloaded

### Pretrained Assets

Downloaded under:

```text
<weights directory>/DeepBranchAI_Zenodo/
```

Installed into:

```text
<nnUNet_results>/Dataset4005_Mitochondria/
```

Default installed checkpoint:

```text
<nnUNet_results>/Dataset4005_Mitochondria/
  nnUNetTrainer_100epochs__nnUNetPlans__3d_fullres/
    fold_2/
      checkpoint_best.pth
```

### VESSEL12 Training Archive

Archive location:

```text
<data directory>/DeepBranchAI_VESSEL12_training.zip
```

Extracted training root:

```text
<data directory>/DeepBranchAI_VESSEL12_training/
```

Installed into nnU-Net raw layout:

```text
<nnUNet_raw>/Dataset3005_Mitochondria/
├─ imagesTr/
├─ labelsTr/
└─ dataset.json
```

### Demo/Test Archive

Archive location:

```text
<data directory>/DeepBranchAI_demo_data.zip
```

Extracted under:

```text
<data directory>/DeepBranchAI_demo_data/
```

## What Gets Staged For The Notebook

After download/install, one helper stages files into the custom-data workflow:

```text
<data directory>/custom_finetune/
├─ raw/
├─ ground_truth/
└─ predict/
```

Specifically:

- one matched VESSEL12 training pair goes into `raw/` and `ground_truth/`
- one demo raw volume goes into `predict/`

## When To Use Which Mode

Use repo-local storage when:

- the data is small
- you are only testing the notebook

Use an external drive when:

- the training archive is too large for the repo drive
- you want checkpoints and nnU-Net intermediates off the system drive
- you want one stable location like `F:\DeepBranchAI`

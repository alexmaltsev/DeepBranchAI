# Finetune On Custom Data

This guide goes with `demo/Demo_Finetune.ipynb`.

## What The Notebook Does

The notebook is a thin front end over the `deepbranchai` helper package. It can:

1. Set up repo-local or external storage for data, weights, and nnU-Net folders.
2. Run either as a staged VESSEL12 demo or against three user-provided input folders.
3. Copy user raw, mask, and inference files into a managed workspace.
4. Convert raw volumes to single-channel grayscale when possible.
5. Check that masks are usable binary masks and binarize common non-binary masks automatically.
6. Check the train/validation split, patch size, axis sizes, and GPU memory budget before training starts.
5. Convert those volumes into nnU-Net training format.
6. Fine-tune from a pretrained checkpoint.
7. Run the fine-tuned model on new volumes.

## Managed Workspace Layout

All relative paths are resolved under the configured workspace.

```text
<data directory>/
└─ custom_finetune/
   ├─ raw/
   │  ├─ sample01.tif
   │  └─ sample02.tif
   ├─ ground_truth/
   │  ├─ sample01_gt.tif
   │  └─ sample02_gt.tif
   ├─ predict/
   │  └─ new_volume_to_segment.tif
   └─ predictions/
```

Recommended naming rule:

```text
raw/sample01.tif
ground_truth/sample01_gt.tif
```

The validator also recognizes `_mask`, `_label`, `_labels`, and `_seg`, but `_gt` is the default rule used throughout the notebook.

## Supported Volume Types

- `.tif`
- `.tiff`
- `.nii`
- `.nii.gz`
- `.mha`
- `.mhd`

## Quick Start

1. Open `demo/Demo_Finetune.ipynb`.
2. Fill in these four paths in the user settings cell:
   - `WORKSPACE_DIR`
   - `TRAINING_RAW_INPUT_DIR`
   - `TRAINING_GROUND_TRUTH_INPUT_DIR`
   - `INFERENCE_INPUT_DIR`
3. Leave the three input folders as `None` to run the staged VESSEL12 demo.
4. Run the notebook with `Run All`.
5. Read the import summary and preflight summary before training starts.
6. At the end, read the printed `Segmentation TIFFs saved to:` line.

## What Each Notebook Section Does

### Import And Storage Setup

Creates the workspace layout and exposes:

- `REPO_DIR`
- `DATA_DIR`
- `WEIGHTS_DIR`

### User Settings

The notebook asks most users to set only:

- `WORKSPACE_DIR`
- `TRAINING_RAW_INPUT_DIR`
- `TRAINING_GROUND_TRUTH_INPUT_DIR`
- `INFERENCE_INPUT_DIR`

The notebook builds the internal `FinetuneConfig` for you and auto-picks a clean nnU-Net dataset id if the default id is already used on that machine.

### Import Demo Data Or Your Own Folders

If you leave the three input folders as `None`, `download_and_stage_vessel12_train_val_demo(config, ...)` does four things:

1. installs the default pretrained checkpoint
2. installs the VESSEL12 training dataset into `nnUNet_raw`
3. stages one VESSEL12 case into `raw` and `ground_truth` for training
4. stages a second VESSEL12 case into `raw` and `ground_truth` for validation, then copies the validation raw volume into `predict`

If you set your own input folders, `import_user_input_folders(...)` does this instead:

1. copies raw volumes into the managed `raw` folder
2. copies masks into the managed `ground_truth` folder
3. renames masks to the `_gt` rule used by the notebook
4. converts raw volumes to grayscale when possible
5. binarizes common mask formats automatically
6. copies inference volumes into the managed `predict` folder
7. prints the managed prediction output folder

### Check Pairing And Masks

`inspect_custom_dataset(config)` checks:

- a raw file has a matching mask
- the raw and mask are both 3D after channel selection
- their shapes match
- the raw volume is single-channel grayscale after import cleanup
- the raw volume is not constant
- the mask is binary, or can be thresholded if configured
- the mask is not empty
- the mask is not almost entirely foreground

### Training Preflight

`preflight_training_setup(config, report=report)` checks the training setup before nnU-Net starts:

- the train/validation split that will be used
- whether one labeled volume needs to be partitioned into train and validation subvolumes
- whether any axis is smaller than the requested patch size
- whether the patch size should be reduced for the current GPU memory budget
- whether CUDA is available

The preflight returns an adjusted `FinetuneConfig` with the recommended patch size and batch size.

### Prepare nnU-Net Dataset

`prepare_nnunet_dataset(config)` writes:

- `imagesTr/*.nii.gz`
- `labelsTr/*.nii.gz`
- `dataset.json`
- `splits_final.json`

under the configured `nnUNet_raw` and `nnUNet_preprocessed` roots.

Split behavior is automatic when you do not set an explicit train/validation split:

- `1` labeled volume: the notebook partitions that volume into non-overlapping train and validation subvolumes
- `2` to `4` labeled volumes: one whole volume is held out for validation and the rest go to training
- `5+` labeled volumes: an approximate `4/5` train and `1/5` validation split is used

### Finetune

`finetune_model(config)` can:

1. run nnU-Net planning and preprocessing
2. load the pretrained checkpoint
3. train the target dataset
4. run validation

If `config.pretrained_weights` is `None`, the notebook uses the default checkpoint path under `nnUNet_results`.

### Segment New Volumes

`predict_with_finetuned_model(config)`:

1. converts `predict` volumes into nnU-Net input files
2. runs inference
3. writes nnU-Net outputs
4. also writes TIFF masks into `predictions`

The notebook prints the prediction folder path at the end so the user does not need to inspect the config.

## Using Your Own Checkpoint

If the checkpoint is not already installed under `nnUNet_results`, point `config.pretrained_weights` at it directly.

Example:

```python
config = FinetuneConfig(
    storage_dir=Path(r"F:\DeepBranchAI"),
    pretrained_weights=WEIGHTS_DIR / "checkpoint_best.pth",
)
```

Relative checkpoint paths are resolved under:

- `weights/` if they start with `weights/...`
- `nnUNet_results/` if they start with `nnUNet_results/...`
- `weights/` by default otherwise

## Notes

- The demo notebook uses one labeled VESSEL12 case for training and a different labeled VESSEL12 case for validation when `USE_VESSEL12_DEMO = True`.
- The full VESSEL12 fine-tuning pipeline lives under `train/finetune/`.
- For multi-channel 4D data, set `channel_index`.
- The notebook is intentionally small; most behavior lives in `deepbranchai/custom_finetune.py`.

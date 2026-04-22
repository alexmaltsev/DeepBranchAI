# Troubleshooting

## Pretrained Checkpoint Not Found

Symptom:

```text
FileNotFoundError: Pretrained checkpoint not found
```

Check:

1. Did you run the VESSEL12 download/stage block?
2. Is the checkpoint installed under `nnUNet_results`?
3. If not, did you set `config.pretrained_weights`?

Common fix:

```python
config = FinetuneConfig(
    storage_dir=Path(r"F:\DeepBranchAI"),
    pretrained_weights=WEIGHTS_DIR / "checkpoint_best.pth",
)
```

## No 3D Image Files Found In `raw_dir`

Symptom:

```text
No 3D image files found in raw_dir
```

Check:

- files are in the configured `raw` folder
- file extensions are supported
- the files are not nested in an unexpected location

## No Mask Found For A Raw Volume

Symptom:

```text
No mask found for sample01.tif. Recommended mask name: sample01_gt.tif
```

Fix:

Rename the mask to match the raw stem plus `_gt`.

```text
raw/sample01.tif
ground_truth/sample01_gt.tif
```

## Mask Is Not Clearly Binary

Symptom:

```text
mask is not clearly binary
```

Meaning:

- the mask contains more than two values
- or it needs thresholding before training

Fix options:

1. rewrite the mask so it contains only background and foreground
2. set `binary_threshold`

Example:

```python
config = FinetuneConfig(binary_threshold=0.5)
```

## Mask Has No Foreground Voxels

Symptom:

```text
mask has no foreground voxels
```

Fix:

- confirm the correct file was placed in `ground_truth`
- confirm the mask is not all background after thresholding

## Mask Is Mostly Foreground

Symptom:

```text
mask is 95% foreground, which is probably not a valid mask
```

Fix:

- confirm the mask is actually a segmentation mask and not a raw image
- confirm foreground/background coding is correct

## Raw And Mask Shapes Do Not Match

Symptom:

```text
shape mismatch: raw (...), mask (...)
```

Fix:

- resample or export the raw and mask so they match exactly
- verify both files refer to the same volume

The notebook will not continue until shapes match.

## Raw Volume Is Constant

Symptom:

```text
raw volume is constant
```

Meaning:

- the raw file is probably corrupt
- or the wrong file was placed in `raw`

## 4D Multi-Channel Volume Error

Symptom:

```text
appears to be multi-channel
```

Fix:

Set `channel_index` in `FinetuneConfig`.

Example:

```python
config = FinetuneConfig(channel_index=0)
```

## No CUDA GPU Detected

Symptom:

`check_gpu()` prints that CUDA is unavailable.

Meaning:

- the workflow can still run, but training and inference will be much slower

Check:

- CUDA-enabled PyTorch is installed
- the correct environment is active
- the GPU driver is available

## nnU-Net Planning Or Preprocessing Fails

Check:

1. `nnUNetv2_plan_and_preprocess` is installed
2. the environment variables point to the expected storage roots
3. `dataset.json`, `imagesTr`, and `labelsTr` were created
4. masks are valid and non-empty

Useful paths to inspect:

```text
<nnUNet_raw>/Dataset<id>_<name>/
<nnUNet_preprocessed>/Dataset<id>_<name>/
```

## Fine-Tuning Starts But Uses The Wrong Checkpoint

Check:

- `default_pretrained_weights(config)` output
- `config.pretrained_weights`
- `storage_dir` and `weights_dir`

Relative checkpoint paths may resolve differently depending on whether they begin with:

- `weights/...`
- `nnUNet_results/...`

## Prediction Folder Is Empty

Check:

1. the `predict` folder contains at least one supported volume
2. the fine-tuned model checkpoint exists
3. the configured dataset id, fold, trainer, and configuration match the trained output

## Where Outputs Go

Prediction outputs are written to:

```text
<data directory>/custom_finetune/predictions/
```

That folder contains:

- nnU-Net output volumes
- TIFF masks exported from the nnU-Net outputs

# DeepBranchAI Model Card

## Model Summary

DeepBranchAI is a 3D nnU-Net checkpoint trained to segment the mitochondrial reticulum in skeletal-muscle FIB-SEM volumes. The source task contains dense, heterogeneous, continuous 3D branching structure. The released weights are intended to initialize supervised fine-tuning on other branching-network datasets.

DeepBranchAI is not presented as a universal zero-shot segmenter. The accepted paper evaluates whether its initialization improves fine-tuning on paired image/reference-label datasets from venous CT, plant-root CT, and airway CT.

## Architecture And Training

| Property | Value |
|---|---|
| Framework | nnU-Net v2.3.1, 3D full-resolution configuration |
| Source images | Skeletal-muscle mitochondrial FIB-SEM |
| Source voxel size | 15 nm isotropic |
| Released patch size | 352 x 352 x 128 voxels |
| Source volumes | 20 full 128-slice volumes |
| Validation | Subject-grouped five-fold cross-validation |
| Training objective | nnU-Net cross-entropy plus Dice loss |
| Training schedule | 100 epochs |
| Output | Binary foreground/background segmentation |
| Threshold | 0.50 |

Source labels were created from machine-generated drafts and manually refined by experts before 3D model training. Evaluation metrics were computed volume-wise on full held-out predictions after nnU-Net prediction assembly.

## Released Weights

The five cross-validation checkpoints and configuration files are hosted on [Zenodo](https://zenodo.org/records/19363534). Fold 0 is the default for single-checkpoint transfer because it initialized the external fine-tuning experiments in the accepted paper.

| Fold | Direct download |
|---:|---|
| 0 | [checkpoint](https://zenodo.org/records/19363534/files/DeepBranchAI_MitoEye_fold0.pth?download=1) |
| 1 | [checkpoint](https://zenodo.org/records/19363534/files/DeepBranchAI_MitoEye_fold1.pth?download=1) |
| 2 | [checkpoint](https://zenodo.org/records/19363534/files/DeepBranchAI_MitoEye_fold2.pth?download=1) |
| 3 | [checkpoint](https://zenodo.org/records/19363534/files/DeepBranchAI_MitoEye_fold3.pth?download=1) |
| 4 | [checkpoint](https://zenodo.org/records/19363534/files/DeepBranchAI_MitoEye_fold4.pth?download=1) |

## Evaluation Summary

Source-domain five-fold performance was Dice `0.942 +/- 0.020` and clDice `0.940 +/- 0.021`. External fine-tuning was evaluated on 3D-IRCADb venous-system CT (`n=12`), Plant CT roots (`n=29`), and AeroPath airway CT (`n=27`). DeepBranchAI-pretrained fine-tuning had higher mean clDice than scratch nnU-Net in all three datasets. Full means, standard deviations, comparator results, and paired tests are in [`results/manuscript/`](results/manuscript/).

## Intended Use

- Initialization for supervised fine-tuning on 3D branching or tubular structures.
- Research on continuity-sensitive volumetric segmentation.
- Comparison of source-domain pretraining for 3D segmentation.

Users should retain held-out target-domain reference labels and report both overlap and continuity-sensitive measures. The repository provides Dice, clDice, and 26-connected absolute connected-component error implementations.

## Limitations

- Transfer has been evaluated on three external CT datasets, not every imaging modality or branching structure.
- The model outputs a binary mask and does not assign branch identities or instances.
- Fine-tuning performance depends on target data quality, label definitions, preprocessing, and training budget.
- clDice and component-count error measure continuity but do not replace downstream network measurements such as branch length, junction count, tortuosity, or caliber.
- The source and external datasets are modest in size; fold dispersion and paired-volume statistics should be considered with the reported means.

## License

Repository materials are CC0-1.0. Source datasets and third-party comparator checkpoints retain their original terms.

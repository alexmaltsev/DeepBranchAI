# DeepBranchAI

**A Novel Cascade Workflow Enabling Accessible 3D Branching Network Segmentation**

Alexander V. Maltsev†, Lisa M. Hartnell†, Luigi Ferrucci\*
*Intramural Research Program, National Institute on Aging, Baltimore, MD, United States*
†*These authors contributed equally to this work and share first authorship.*

---

## Overview

DeepBranchAI is a 3D nnU-Net model optimized for topology-preserving segmentation of branching networks. It is trained through a cascade workflow that combines conventional machine learning, deep learning, and expert refinement to overcome the annotation bottleneck inherent in 3D volumetric segmentation.

Three-dimensional branching networks — mitochondria, vasculature, root systems, porous materials, neural circuits — share a common vulnerability: minor voxel misclassifications can break or amplify connectivity, distorting the network's true topology. Accurate segmentation requires volumetric (3D) models, but training those models demands far more annotated data than 2D approaches. DeepBranchAI addresses this through a cascade training framework that transforms sparse initial labels into comprehensive training sets, reducing annotation time from months to weeks.

### Key Results

- **DSC = 0.942** across 5-fold cross-validation on FIB-SEM mitochondrial networks (15 nm isotropic voxels)
- **97.05% accuracy** on VESSEL12 lung vasculature (CT volumes) via transfer learning using only 10% of target data
- Successful transfer across a **30,000-fold voxel size difference**, different imaging modalities, and different biological systems

## Cascade Training Framework

The workflow proceeds through three stages:

**Stage A — Preprocessing & Training Set Curation:** FIB-SEM volumes are aligned, denoised, and curated for topological diversity with a minimum depth of 128 Z-slices and balanced class representation.

**Stage B — Iterative Ground Truth Generation:** Initial segmentation begins with 2D Weka random forests trained on minimal annotations (~5–10 minutes). Experts correct outputs, retrain iteratively until accuracy plateaus, then transition to a 2D nnU-Net. The trained 2D model generates probability maps that experts refine into ground truth for the 3D model. Each cycle produces better drafts, creating a positive feedback loop.

**Stage C — 3D nnU-Net Training (DeepBranchAI):** The final model trains on 360×360×128 voxel patches with 5-fold cross-validation (100 epochs). Inference uses overlapping 3D patches with stitching and weighted averaging, thresholded at 0.50.

## Performance

| Model | Sensitivity | Specificity | Accuracy | DSC | AVD (%) | κ |
|---|---|---|---|---|---|---|
| 2D U-Net | 0.721 | 0.971 | 0.947 | 0.726 | 9.26 | 0.697 |
| 3D U-Net | 0.856 | 0.993 | 0.978 | 0.888 | 8.86 | 0.875 |
| 2D nnU-Net | 0.880 | 0.986 | 0.976 | 0.879 | 8.87 | 0.865 |
| **DeepBranchAI (3D nnU-Net)** | **0.925** | **0.996** | **0.989** | **0.942** | **6.04** | **0.935** |

## Domain Applications

The cascade workflow generalizes to any domain where 3D connectivity must be preserved:

| Domain | Target Structure | Imaging Modality |
|---|---|---|
| Cell Biology | Mitochondrial networks | FIB-SEM |
| Vascular Biology | Vascular networks | CT |
| Neuroscience | Neural circuits | EM volumes |
| Materials Science | Porous membranes | Micro-CT |
| Geophysics | Fracture networks | Seismic volumes |
| Plant Biology | Root systems | MRI/CT |
| Engineering | 3D printed lattices | X-ray CT |

## Repository Contents

> **Note:** Full code and trained weights are being prepared for upload. This repository currently serves as a placeholder accompanying the manuscript.

### Notebooks (coming soon)

| Notebook | Purpose |
|---|---|
| `Destriping.ipynb` | Wavelet-FFT filtering for stripe artifact removal |
| `TIFF_Analysis.ipynb` | Quality control, dimension verification, intensity statistics |
| `Multi-fold_Segmentation_2D.ipynb` | 2D model evaluation metrics (IoU, Dice, precision, recall) |
| `Multi-fold_Segmentation_3D.ipynb` | 3D model evaluation with volumetric analysis |
| `VESSEL12_Analysis.ipynb` | Transfer learning validation pipeline |
| `Parametric_Weight_Analysis.ipynb` | Network weight analysis and hyperparameter exploration |
| `Processing_Functions.ipynb` | Helper functions for data manipulation and batch processing |

### Pre-trained Weights (coming soon)

Trained model weights in nnU-Net v2.3.1 format for immediate deployment or fine-tuning.

## Requirements

- Python 3.12.2
- PyTorch 2.2.1 with CUDA 11.8
- nnU-Net v2.3.1
- nibabel, tifffile, numpy, multiprocessing
- [aind-smartspim-destripe](https://github.com/AllenNeuralDynamics/aind-smartspim-destripe) (for denoising)
- [ORS Dragonfly 4.0](https://dragonfly.comet.tech/) (for manual annotation and refinement)
- [ImageJ/Fiji](https://fiji.sc/) with Trainable Weka Segmentation plugin (for initial 2D segmentation)

### Hardware

Training was performed on an NVIDIA RTX A6000 (48 GB VRAM), 128 GB RAM, 24 logical processors, Ubuntu 24. The 48 GB VRAM accommodates 360×360×128 voxel patches; 24 GB VRAM accommodates approximately 50% of this volume. nnU-Net automatically adapts to available memory.

### Environment Setup

Before training, configure nnU-Net environment variables:

```bash
export nnUNet_raw="$PWD/nnUNet_raw"
export nnUNet_preprocessed="$PWD/nnUNet_preprocessed"
export nnUNet_results="$PWD/nnUNet_results"
```

## Citation

If you use DeepBranchAI or the cascade training workflow in your research, please cite:

```
Maltsev, A.V.*, Hartnell, L.M.*, Ferrucci, L. DeepBranchAI: A Novel Cascade Workflow
Enabling Accessible 3D Branching Network Segmentation. [Preprint/Journal TBD].
```

## License

Released under the CC0 license.

## Funding

This research was supported by the Intramural Research Program of the National Institutes of Health (NIH). The findings and conclusions presented in this paper are those of the authors and do not necessarily reflect the views of the NIH or the U.S. Department of Health and Human Services.

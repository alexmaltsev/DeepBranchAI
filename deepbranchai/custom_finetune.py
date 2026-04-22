"""Custom-data finetuning workflow helpers."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd

from .image_io import (
    SUPPORTED_EXTENSIONS,
    list_volume_files,
    load_volume,
    save_nifti,
    save_tiff,
    validate_binary_mask,
    volume_stem,
)
from .downloads import (
    download_and_install_pretrained_weights,
    download_and_install_vessel12_training_data,
    download_vessel12_reference_plans,
    download_vessel12_demo_data,
    find_vessel12_demo_volume,
)
from .nnunet_runner import (
    dataset_folder_name,
    extract_fingerprints_only,
    finetune_from_pretrained,
    preprocess_with_existing_plans,
    predict_nnunet,
    run_planning_and_preprocessing,
)
from .paths import setup_environment
from .visualization import render_figure


DEFAULT_MASK_SUFFIXES = ("_gt", "_mask", "_label", "_labels", "_seg")


def _strip_named_root(path: Path, root_name: str) -> Path:
    parts = path.parts
    if parts and parts[0].lower() == root_name.lower():
        return Path(*parts[1:]) if len(parts) > 1 else Path()
    return path


def _resolve_under_root(value: str | Path, root: Path, root_name: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    path = _strip_named_root(path, root_name)
    return (root / path).resolve()


def _resolve_checkpoint_path(value: str | Path, paths: dict[str, Path]) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()

    first = path.parts[0].lower() if path.parts else ""
    if first == "weights":
        return (paths["weights"] / _strip_named_root(path, "weights")).resolve()
    if first == "nnunet_results":
        return (paths["nnUNet_results"] / _strip_named_root(path, "nnUNet_results")).resolve()
    return (paths["weights"] / path).resolve()


def _resolve_plans_path(value: str | Path, paths: dict[str, Path]) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()

    first = path.parts[0].lower() if path.parts else ""
    if first == "weights":
        return (paths["weights"] / _strip_named_root(path, "weights")).resolve()
    if first == "nnunet_preprocessed":
        return (paths["nnUNet_preprocessed"] / _strip_named_root(path, "nnUNet_preprocessed")).resolve()
    return (paths["weights"] / path).resolve()


@dataclass
class FinetuneConfig:
    """Configuration for the user-data finetuning template."""

    base_dir: Path | str | None = None
    storage_dir: Path | str | None = None
    data_dir: Path | str | None = None
    weights_dir: Path | str | None = None
    tmp_dir: Path | str | None = None
    nnunet_raw: Path | str | None = None
    nnunet_preprocessed: Path | str | None = None
    nnunet_results: Path | str | None = None
    project_name: str = "custom_branching_data"
    dataset_id: int = 3010
    raw_dir: Path | str = Path("custom_finetune/raw")
    ground_truth_dir: Path | str = Path("custom_finetune/ground_truth")
    predict_dir: Path | str = Path("custom_finetune/predict")
    output_dir: Path | str = Path("custom_finetune/predictions")
    ground_truth_suffix: str = "_gt"
    accepted_mask_suffixes: tuple[str, ...] = DEFAULT_MASK_SUFFIXES
    binary_threshold: float | None = None
    channel_index: int | None = None
    fold: int = 0
    max_epochs: int = 10
    trainer: str = "nnUNetTrainer_100epochs"
    plans: str = "nnUNetPlans"
    configuration: str = "3d_fullres"
    pretrained_weights: Path | str | None = None
    pretrained_dataset_name: str = "Dataset4005_Mitochondria"
    pretrained_trainer_dir: str = "nnUNetTrainer_100epochs__nnUNetPlans__3d_fullres"
    pretrained_fold: int = 2
    train_case_names: tuple[str, ...] | None = None
    validation_case_names: tuple[str, ...] | None = None
    reference_plans: Path | str | None = None
    reference_patch_size: tuple[int, ...] | None = None
    reference_batch_size: int | None = None
    overwrite_dataset: bool = True
    run_preprocessing: bool = True
    preprocess_configurations: tuple[str, ...] = ("3d_fullres",)
    num_processes: int = 2
    reuse_existing_training: bool = True

    def environment_paths(self, base_dir: str | Path | None = None, verbose: bool = False) -> dict[str, Path]:
        """Resolve storage roots and configure nnU-Net environment variables."""
        return setup_environment(
            base_dir=base_dir or self.base_dir,
            storage_dir=self.storage_dir,
            data_dir=self.data_dir,
            weights_dir=self.weights_dir,
            tmp_dir=self.tmp_dir,
            nnunet_raw=self.nnunet_raw,
            nnunet_preprocessed=self.nnunet_preprocessed,
            nnunet_results=self.nnunet_results,
            verbose=verbose,
        )

    def resolve(self, base_dir: str | Path | None = None) -> "FinetuneConfig":
        paths = self.environment_paths(base_dir, verbose=False)
        resolved = FinetuneConfig(**self.__dict__)
        resolved.base_dir = paths["base"]
        resolved.storage_dir = paths["storage"]
        resolved.data_dir = paths["data"]
        resolved.weights_dir = paths["weights"]
        resolved.tmp_dir = paths["tmp"]
        resolved.nnunet_raw = paths["nnUNet_raw"]
        resolved.nnunet_preprocessed = paths["nnUNet_preprocessed"]
        resolved.nnunet_results = paths["nnUNet_results"]
        for attr in ("raw_dir", "ground_truth_dir", "predict_dir", "output_dir"):
            setattr(resolved, attr, _resolve_under_root(getattr(resolved, attr), paths["data"], "data"))
        if resolved.pretrained_weights is not None:
            resolved.pretrained_weights = _resolve_checkpoint_path(resolved.pretrained_weights, paths)
        if resolved.reference_plans is not None:
            resolved.reference_plans = _resolve_plans_path(resolved.reference_plans, paths)
        return resolved


@dataclass(frozen=True)
class VolumePair:
    case_id: str
    raw_path: Path
    mask_path: Path


@dataclass
class UserInputImportReport:
    staged_pairs: list[VolumePair] = field(default_factory=list)
    staged_predict_inputs: list[Path] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    raw_dir: Path | None = None
    ground_truth_dir: Path | None = None
    predict_dir: Path | None = None
    output_dir: Path | None = None

    def show(self) -> None:
        print(f"Staged training pairs: {len(self.staged_pairs)}")
        print(f"Staged inference volumes: {len(self.staged_predict_inputs)}")
        if self.raw_dir is not None:
            print(f"Working raw dir:         {self.raw_dir}")
        if self.ground_truth_dir is not None:
            print(f"Working ground truth:    {self.ground_truth_dir}")
        if self.predict_dir is not None:
            print(f"Working inference dir:   {self.predict_dir}")
        if self.output_dir is not None:
            print(f"Segmentation output dir: {self.output_dir}")
        if self.errors:
            print("\nBlocking issues:")
            for message in self.errors:
                print(f"  - {message}")
        if self.warnings:
            print("\nWarnings:")
            for message in self.warnings:
                print(f"  - {message}")

    def raise_if_blocking_errors(self) -> None:
        if self.errors:
            joined = "\n".join(f"- {message}" for message in self.errors)
            raise ValueError(f"Fix these input import issues before continuing:\n{joined}")


@dataclass
class DatasetInspectionReport:
    pairs: list[VolumePair] = field(default_factory=list)
    rows: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    unmatched_raw: list[Path] = field(default_factory=list)
    unmatched_masks: list[Path] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors and bool(self.pairs)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)

    def show(self) -> pd.DataFrame:
        print(f"Matched training pairs: {len(self.pairs)}")
        if self.errors:
            print("\nBlocking issues:")
            for message in self.errors:
                print(f"  - {message}")
        if self.warnings:
            print("\nWarnings:")
            for message in self.warnings:
                print(f"  - {message}")
        df = self.to_dataframe()
        if not df.empty:
            try:
                display(df)  # noqa: F821 - provided by notebooks
            except NameError:
                print(df.to_string(index=False))
        return df

    def raise_if_blocking_errors(self) -> None:
        if self.errors:
            joined = "\n".join(f"- {message}" for message in self.errors)
            raise ValueError(f"Fix these data setup issues before continuing:\n{joined}")
        if not self.pairs:
            raise ValueError("No valid raw/ground-truth pairs were found.")


@dataclass
class TrainingPreflightReport:
    config: FinetuneConfig
    source_case_count: int
    split_mode: str
    train_source_case_ids: list[str]
    validation_source_case_ids: list[str]
    original_patch_size: tuple[int, ...]
    recommended_patch_size: tuple[int, ...]
    recommended_batch_size: int
    train_shape_min: tuple[int, ...] | None
    train_shape_max: tuple[int, ...] | None
    gpu_available: bool
    gpu_name: str | None = None
    gpu_vram_gb: float | None = None
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def show(self) -> None:
        print("Training preflight")
        print(f"  source cases:        {self.source_case_count}")
        print(f"  split mode:          {self.split_mode}")
        print(f"  train cases:         {self.train_source_case_ids}")
        print(f"  validation cases:    {self.validation_source_case_ids}")
        print(f"  patch size:          {self.original_patch_size} -> {self.recommended_patch_size}")
        print(f"  batch size:          {self.recommended_batch_size}")
        if self.train_shape_min is not None:
            print(f"  train shape min:     {self.train_shape_min}")
        if self.train_shape_max is not None:
            print(f"  train shape max:     {self.train_shape_max}")
        if self.gpu_available:
            print(f"  GPU:                 {self.gpu_name} ({self.gpu_vram_gb:.1f} GB)")
        else:
            print("  GPU:                 none detected")
        if self.errors:
            print("\nBlocking issues:")
            for message in self.errors:
                print(f"  - {message}")
        if self.warnings:
            print("\nWarnings:")
            for message in self.warnings:
                print(f"  - {message}")
        if self.notes:
            print("\nNotes:")
            for message in self.notes:
                print(f"  - {message}")

    def raise_if_blocking_errors(self) -> None:
        if self.errors:
            joined = "\n".join(f"- {message}" for message in self.errors)
            raise ValueError(f"Fix these training preflight issues before continuing:\n{joined}")


def create_custom_finetune_folders(config: FinetuneConfig, base_dir: str | Path | None = None) -> FinetuneConfig:
    """Create raw, ground_truth, predict, and output folders for the template."""
    config = config.resolve(base_dir)
    for path in (config.raw_dir, config.ground_truth_dir, config.predict_dir, config.output_dir):
        Path(path).mkdir(parents=True, exist_ok=True)
    return config


def _read_volume_any(path: str | Path) -> np.ndarray:
    path = Path(path)
    name = path.name.lower()
    if name.endswith((".tif", ".tiff")):
        import tifffile

        arr = tifffile.imread(str(path))
    elif name.endswith((".nii", ".nii.gz")):
        arr = np.asanyarray(nib.load(str(path)).dataobj)
    elif name.endswith((".mha", ".mhd")):
        try:
            import SimpleITK as sitk
        except ImportError as exc:
            raise ImportError("SimpleITK is required to read MHA/MHD files") from exc
        arr = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
    else:
        raise ValueError(f"Unsupported volume file: {path}")
    arr = np.asarray(arr)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{path.name} contains NaN or infinite values")
    return arr


def _find_channel_axis(arr: np.ndarray) -> int | None:
    if arr.ndim != 4:
        return None
    candidates = [axis for axis, size in enumerate(arr.shape) if size <= 4]
    if not candidates:
        return None
    for axis in (arr.ndim - 1, 1, 0):
        if axis in candidates:
            return axis
    return candidates[0]


def _sanitize_raw_volume(arr: np.ndarray, path: Path) -> tuple[np.ndarray, list[str]]:
    warnings: list[str] = []
    if arr.ndim == 3:
        return arr, warnings
    if arr.ndim != 4:
        raise ValueError(f"{path.name} must be a 3D volume or a 4D volume with a channel axis; got shape {arr.shape}")

    channel_axis = _find_channel_axis(arr)
    if channel_axis is None:
        raise ValueError(
            f"{path.name} looks multi-channel with shape {arr.shape}, but the channel axis could not be inferred."
        )

    channels = np.moveaxis(arr, channel_axis, -1)
    n_channels = channels.shape[-1]
    if n_channels == 1:
        warnings.append(f"{path.name}: squeezed single-channel axis {channel_axis}.")
        return np.asarray(channels[..., 0]), warnings
    if n_channels in (3, 4):
        rgb = channels[..., :3].astype(np.float32)
        gray = (0.2989 * rgb[..., 0]) + (0.5870 * rgb[..., 1]) + (0.1140 * rgb[..., 2])
        warnings.append(f"{path.name}: converted {n_channels}-channel volume to grayscale.")
        return gray, warnings
    raise ValueError(
        f"{path.name} has {n_channels} channels on axis {channel_axis}. Provide a single-channel grayscale volume."
    )


def _sanitize_mask_volume(arr: np.ndarray, path: Path) -> tuple[np.ndarray, list[str]]:
    warnings: list[str] = []
    if arr.ndim == 4:
        channel_axis = _find_channel_axis(arr)
        if channel_axis is None:
            raise ValueError(
                f"{path.name} looks multi-channel with shape {arr.shape}, but the mask channel axis could not be inferred."
            )
        channels = np.moveaxis(arr, channel_axis, -1)
        n_channels = channels.shape[-1]
        if n_channels == 1:
            warnings.append(f"{path.name}: squeezed single-channel axis {channel_axis}.")
            arr = np.asarray(channels[..., 0])
        elif n_channels in (3, 4):
            first = channels[..., 0]
            if np.allclose(channels[..., :3], np.expand_dims(first, axis=-1)):
                arr = np.asarray(first)
                warnings.append(f"{path.name}: used the first of identical mask channels.")
            else:
                arr = np.max(channels[..., :3], axis=-1)
                warnings.append(f"{path.name}: collapsed multi-channel mask by taking the channel-wise maximum.")
        else:
            raise ValueError(f"{path.name} has {n_channels} channels in the mask. Provide a single-channel mask.")

    if arr.ndim != 3:
        raise ValueError(f"{path.name} must be a 3D volume after channel cleanup; got shape {arr.shape}")

    unique = np.unique(arr)
    if unique.size > 2:
        if np.issubdtype(arr.dtype, np.floating) and float(np.min(arr)) >= 0.0 and float(np.max(arr)) <= 1.0:
            arr = (arr >= 0.5).astype(np.uint8)
            warnings.append(f"{path.name}: thresholded floating mask at >= 0.5.")
        else:
            arr = (arr > 0).astype(np.uint8)
            warnings.append(f"{path.name}: converted non-binary mask to foreground > 0.")

    check = validate_binary_mask(arr, None)
    warnings.extend(check.warnings)
    return check.mask, warnings


def _pair_input_training_files(raw_files: list[Path], mask_files: list[Path], suffixes: tuple[str, ...]) -> DatasetInspectionReport:
    report = DatasetInspectionReport()
    masks_by_stem: dict[str, list[Path]] = {}
    for path in mask_files:
        masks_by_stem.setdefault(volume_stem(path), []).append(path)

    used_masks: set[Path] = set()
    for raw_path in raw_files:
        raw_stem = volume_stem(raw_path)
        candidate_stems = [raw_stem] + [f"{raw_stem}{suffix}" for suffix in suffixes]
        match = None
        for stem in candidate_stems:
            candidates = [path for path in masks_by_stem.get(stem, []) if path not in used_masks]
            if len(candidates) == 1:
                match = candidates[0]
                break
        if match is None:
            expected = f"{raw_stem}_gt{raw_path.suffix if raw_path.suffix.lower() != '.gz' else '.nii.gz'}"
            report.errors.append(
                f"No ground-truth mask found for {raw_path.name}. Put the matching mask in the mask input folder and rename it like {expected}."
            )
            report.unmatched_raw.append(raw_path)
            continue
        used_masks.add(match)
        report.pairs.append(VolumePair(raw_stem, raw_path, match))

    report.unmatched_masks = [path for path in mask_files if path not in used_masks]
    for path in report.unmatched_masks:
        report.warnings.append(f"Mask input was not used because no raw file matched it: {path.name}")
    return report


def _clear_directory_contents(path: Path) -> None:
    for child in path.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def reset_custom_finetune_folders(
    config: FinetuneConfig,
    base_dir: str | Path | None = None,
    clear_output_dir: bool = True,
) -> FinetuneConfig:
    """Clear demo folders so notebook Run All starts from a known state."""
    config = create_custom_finetune_folders(config, base_dir)
    targets = [config.raw_dir, config.ground_truth_dir, config.predict_dir]
    if clear_output_dir:
        targets.append(config.output_dir)
    for path in targets:
        _clear_directory_contents(Path(path))
    return config


def import_user_input_folders(
    config: FinetuneConfig,
    training_raw_input_dir: str | Path,
    training_ground_truth_input_dir: str | Path,
    inference_input_dir: str | Path | None = None,
    base_dir: str | Path | None = None,
    reset_workspace: bool = True,
) -> UserInputImportReport:
    """Copy user-provided folders into the managed workspace, fixing common raw/mask issues on the way."""
    config = create_custom_finetune_folders(config, base_dir)
    if reset_workspace:
        config = reset_custom_finetune_folders(config, base_dir, clear_output_dir=True)

    raw_input_dir = Path(training_raw_input_dir)
    ground_truth_input_dir = Path(training_ground_truth_input_dir)
    report = UserInputImportReport(
        raw_dir=Path(config.raw_dir),
        ground_truth_dir=Path(config.ground_truth_dir),
        predict_dir=Path(config.predict_dir),
        output_dir=Path(config.output_dir),
    )

    if not raw_input_dir.exists():
        report.errors.append(f"Training raw input folder does not exist: {raw_input_dir}")
    if not ground_truth_input_dir.exists():
        report.errors.append(f"Training ground-truth input folder does not exist: {ground_truth_input_dir}")
    if report.errors:
        return report

    raw_files = list_volume_files(raw_input_dir)
    mask_files = list_volume_files(ground_truth_input_dir)
    if not raw_files:
        report.errors.append(f"No 3D volume files found in the training raw input folder: {raw_input_dir}")
    if not mask_files:
        report.errors.append(f"No 3D volume files found in the training ground-truth input folder: {ground_truth_input_dir}")
    if report.errors:
        return report

    pair_report = _pair_input_training_files(raw_files, mask_files, (config.ground_truth_suffix,) + tuple(
        suffix for suffix in config.accepted_mask_suffixes if suffix != config.ground_truth_suffix
    ))
    report.errors.extend(pair_report.errors)
    report.warnings.extend(pair_report.warnings)
    if report.errors:
        return report

    for pair in pair_report.pairs:
        try:
            raw_array, raw_warnings = _sanitize_raw_volume(_read_volume_any(pair.raw_path), pair.raw_path)
            mask_array, mask_warnings = _sanitize_mask_volume(_read_volume_any(pair.mask_path), pair.mask_path)
            if raw_array.shape != mask_array.shape:
                raise ValueError(
                    f"{pair.raw_path.name} and {pair.mask_path.name} do not match after cleanup: {raw_array.shape} vs {mask_array.shape}"
                )
            raw_dst = Path(config.raw_dir) / f"{pair.case_id}.tif"
            mask_dst = Path(config.ground_truth_dir) / f"{pair.case_id}{config.ground_truth_suffix}.tif"
            save_tiff(raw_array, raw_dst)
            save_tiff(mask_array.astype(np.uint8), mask_dst)
            report.staged_pairs.append(VolumePair(pair.case_id, raw_dst, mask_dst))
            report.warnings.extend(raw_warnings)
            report.warnings.extend(mask_warnings)
        except Exception as exc:
            report.errors.append(f"{pair.raw_path.name} / {pair.mask_path.name}: {exc}")

    inference_source_dir = Path(inference_input_dir) if inference_input_dir is not None else raw_input_dir
    if inference_input_dir is None:
        report.warnings.append("Inference input folder was not set. The raw input folder will also be used for inference.")
    if not inference_source_dir.exists():
        report.errors.append(f"Inference input folder does not exist: {inference_source_dir}")
        return report

    inference_files = list_volume_files(inference_source_dir)
    if not inference_files:
        report.errors.append(f"No 3D volume files found in the inference input folder: {inference_source_dir}")
        return report

    for path in inference_files:
        try:
            raw_array, raw_warnings = _sanitize_raw_volume(_read_volume_any(path), path)
            dst = Path(config.predict_dir) / f"{volume_stem(path)}.tif"
            save_tiff(raw_array, dst)
            report.staged_predict_inputs.append(dst)
            report.warnings.extend(raw_warnings)
        except Exception as exc:
            report.errors.append(f"{path.name}: {exc}")

    return report


def stage_predict_volume(
    config: FinetuneConfig,
    source_path: str | Path,
    base_dir: str | Path | None = None,
    target_name: str | None = None,
    overwrite: bool = False,
) -> Path:
    """Copy one volume into the predict folder."""
    config = create_custom_finetune_folders(config, base_dir)
    source_path = Path(source_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Prediction source not found: {source_path}")

    if target_name is None:
        target_name = source_path.name
    target_path = config.predict_dir / target_name
    if overwrite or not target_path.exists():
        shutil.copy2(str(source_path), str(target_path))
    print(f"Staged predict volume: {target_path}")
    return target_path


def _candidate_mask_names(raw_path: Path, suffixes: tuple[str, ...]) -> list[str]:
    stem = volume_stem(raw_path)
    names: list[str] = []
    for suffix in suffixes:
        for ext in (".tif", ".tiff", ".nii.gz", ".nii", ".mha", ".mhd"):
            names.append(f"{stem}{suffix}{ext}")
    return names


def _safe_case_id(name: str) -> str:
    safe = "".join(ch if ch.isalnum() else "_" for ch in name).strip("_")
    return safe or "case"


def _volume_extension(path: Path) -> str:
    name = path.name.lower()
    for ext in sorted(SUPPORTED_EXTENSIONS, key=len, reverse=True):
        if name.endswith(ext):
            return ext
    return path.suffix


def _nnunet_case_id_from_image(path: Path) -> str:
    stem = volume_stem(path)
    return stem[:-5] if stem.endswith("_0000") else stem


def _find_nnunet_training_pairs(dataset_dir: Path) -> list[tuple[str, Path, Path]]:
    images_dir = dataset_dir / "imagesTr"
    labels_dir = dataset_dir / "labelsTr"
    if not images_dir.exists() or not labels_dir.exists():
        return []

    labels_by_case = {volume_stem(path): path for path in list_volume_files(labels_dir)}
    pairs: list[tuple[str, Path, Path]] = []
    for image_path in list_volume_files(images_dir):
        case_id = _nnunet_case_id_from_image(image_path)
        mask_path = labels_by_case.get(case_id)
        if mask_path is not None:
            pairs.append((case_id, image_path, mask_path))
    return pairs


def _vessel12_dataset_candidates(paths: dict[str, Path]) -> list[Path]:
    candidates = [
        paths["nnUNet_raw"] / "Dataset3005_Mitochondria",
        paths["data"] / "DeepBranchAI_VESSEL12_training",
        paths["data"] / "Dataset3005_Mitochondria",
    ]
    candidates.extend(path.parent for path in paths["data"].rglob("imagesTr"))
    return list(dict.fromkeys(candidates))


def _find_vessel12_training_pair(paths: dict[str, Path], case_id: str | None = None) -> tuple[str, Path, Path] | None:
    selected: tuple[str, Path, Path] | None = None
    for dataset_dir in _vessel12_dataset_candidates(paths):
        for pair in _find_nnunet_training_pairs(dataset_dir):
            if case_id is None or pair[0] == case_id:
                selected = pair
                break
        if selected is not None:
            break
    return selected


def stage_vessel12_example(
    config: FinetuneConfig,
    base_dir: str | Path | None = None,
    case_id: str | None = None,
    overwrite: bool = False,
    stage_predict_copy: bool = True,
) -> VolumePair | None:
    """Copy one installed VESSEL12 raw/mask pair into the custom-data folders."""
    paths = config.environment_paths(base_dir, verbose=False)
    config = create_custom_finetune_folders(config, paths["base"])

    existing_raw = list_volume_files(config.raw_dir)
    existing_masks = list_volume_files(config.ground_truth_dir)
    if not overwrite and (existing_raw or existing_masks):
        print("Custom finetune folders already contain files; leaving them unchanged.")
        return None

    selected = _find_vessel12_training_pair(paths, case_id=case_id)
    if selected is None:
        print("VESSEL12 training data was not found in the configured storage folders.")
        print("Put your own 3D volume in:")
        print(f"  {config.raw_dir}")
        print("Put the matching mask in:")
        print(f"  {config.ground_truth_dir}")
        print(f"Name the mask like raw_name{config.ground_truth_suffix}.tif")
        return None

    found_case_id, raw_src, mask_src = selected
    raw_dst = config.raw_dir / f"{found_case_id}{_volume_extension(raw_src)}"
    mask_dst = config.ground_truth_dir / f"{found_case_id}{config.ground_truth_suffix}{_volume_extension(mask_src)}"

    for src, dst in ((raw_src, raw_dst), (mask_src, mask_dst)):
        if overwrite or not dst.exists():
            shutil.copy2(str(src), str(dst))

    predict_dst = None
    if stage_predict_copy:
        predict_dst = stage_predict_volume(
            config,
            raw_src,
            paths["base"],
            target_name=f"{found_case_id}_to_segment{_volume_extension(raw_src)}",
            overwrite=overwrite,
        )

    print(f"Staged VESSEL12 example case: {found_case_id}")
    print(f"  raw:          {raw_dst}")
    print(f"  ground_truth: {mask_dst}")
    if predict_dst is not None:
        print(f"  predict:      {predict_dst}")
    return VolumePair(found_case_id, raw_dst, mask_dst)


def stage_vessel12_cases(
    config: FinetuneConfig,
    case_ids: list[str] | tuple[str, ...],
    base_dir: str | Path | None = None,
    overwrite: bool = False,
) -> list[VolumePair]:
    """Copy multiple installed VESSEL12 raw/mask pairs into the custom-data folders."""
    paths = config.environment_paths(base_dir, verbose=False)
    config = create_custom_finetune_folders(config, paths["base"])

    staged_pairs: list[VolumePair] = []
    for case_id in case_ids:
        selected = _find_vessel12_training_pair(paths, case_id=case_id)
        if selected is None:
            raise FileNotFoundError(f"VESSEL12 case was not found in the configured storage folders: {case_id}")

        found_case_id, raw_src, mask_src = selected
        raw_dst = config.raw_dir / f"{found_case_id}{_volume_extension(raw_src)}"
        mask_dst = config.ground_truth_dir / f"{found_case_id}{config.ground_truth_suffix}{_volume_extension(mask_src)}"
        for src, dst in ((raw_src, raw_dst), (mask_src, mask_dst)):
            if overwrite or not dst.exists():
                shutil.copy2(str(src), str(dst))

        staged_pairs.append(VolumePair(found_case_id, raw_dst, mask_dst))

    print("Staged VESSEL12 training cases:")
    for pair in staged_pairs:
        print(f"  {pair.case_id}")
    return staged_pairs


def download_and_stage_vessel12_train_val_demo(
    config: FinetuneConfig,
    train_case_id: str,
    validation_case_id: str,
    base_dir: str | Path | None = None,
    overwrite: bool = False,
    reset_demo_folders: bool = False,
) -> dict[str, Path | list[Path] | None]:
    """Install VESSEL12 assets, stage one train case, one validation case, and copy validation raw to predict."""
    paths = config.environment_paths(base_dir, verbose=False)
    config = create_custom_finetune_folders(config, paths["base"])
    if reset_demo_folders:
        config = reset_custom_finetune_folders(config, paths["base"])

    checkpoint = download_and_install_pretrained_weights(paths)
    training_dataset = download_and_install_vessel12_training_data(paths, overwrite=overwrite)
    staged_pairs = stage_vessel12_cases(
        config,
        [train_case_id, validation_case_id],
        base_dir=paths["base"],
        overwrite=overwrite,
    )

    validation_pair = next(pair for pair in staged_pairs if pair.case_id == validation_case_id)
    staged_predict = stage_predict_volume(
        config,
        validation_pair.raw_path,
        paths["base"],
        target_name=validation_pair.raw_path.name,
        overwrite=overwrite,
    )

    return {
        "pretrained_checkpoint": checkpoint,
        "training_dataset": training_dataset,
        "staged_training_raw": next(pair.raw_path for pair in staged_pairs if pair.case_id == train_case_id),
        "staged_training_ground_truth": next(pair.mask_path for pair in staged_pairs if pair.case_id == train_case_id),
        "staged_validation_raw": validation_pair.raw_path,
        "staged_validation_ground_truth": validation_pair.mask_path,
        "staged_predict": staged_predict,
    }


def download_and_stage_vessel12_example(
    config: FinetuneConfig,
    base_dir: str | Path | None = None,
    case_id: str | None = None,
    demo_case_id: str = "VESSEL12_21",
    overwrite: bool = False,
    reset_demo_folders: bool = False,
) -> dict[str, Path | None]:
    """Download VESSEL12 assets, install the pretrained checkpoint, and stage example folders."""
    paths = config.environment_paths(base_dir, verbose=False)
    config = create_custom_finetune_folders(config, paths["base"])
    if reset_demo_folders:
        config = reset_custom_finetune_folders(config, paths["base"])

    checkpoint = download_and_install_pretrained_weights(paths)
    training_dataset = download_and_install_vessel12_training_data(paths, overwrite=overwrite)
    demo_dir = download_vessel12_demo_data(paths)

    staged_pair = stage_vessel12_example(
        config,
        paths["base"],
        case_id=case_id,
        overwrite=overwrite,
        stage_predict_copy=False,
    )

    demo_source = find_vessel12_demo_volume(paths["data"], case_id=demo_case_id)
    staged_predict = None
    if demo_source is not None:
        staged_predict = stage_predict_volume(
            config,
            demo_source,
            paths["base"],
            target_name=demo_source.name,
            overwrite=overwrite,
        )
    elif staged_pair is not None:
        staged_predict = stage_predict_volume(
            config,
            staged_pair.raw_path,
            paths["base"],
            target_name=f"{staged_pair.case_id}_to_segment{_volume_extension(staged_pair.raw_path)}",
            overwrite=overwrite,
        )

    return {
        "pretrained_checkpoint": checkpoint,
        "training_dataset": training_dataset,
        "demo_data": demo_dir,
        "staged_raw": None if staged_pair is None else staged_pair.raw_path,
        "staged_ground_truth": None if staged_pair is None else staged_pair.mask_path,
        "staged_predict": staged_predict,
    }


def find_volume_pairs(config: FinetuneConfig, base_dir: str | Path | None = None) -> DatasetInspectionReport:
    """Match raw volumes to masks using the configured suffix rule."""
    paths = config.environment_paths(base_dir, verbose=False)
    config = config.resolve(paths["base"])
    create_custom_finetune_folders(config, paths["base"])
    report = DatasetInspectionReport()

    raw_files = list_volume_files(config.raw_dir)
    mask_files = list_volume_files(config.ground_truth_dir)
    mask_by_name = {path.name: path for path in mask_files}
    matched_masks: set[Path] = set()

    suffixes = (config.ground_truth_suffix,) + tuple(
        suffix for suffix in config.accepted_mask_suffixes if suffix != config.ground_truth_suffix
    )

    for raw_path in raw_files:
        match = None
        for candidate_name in _candidate_mask_names(raw_path, suffixes):
            if candidate_name in mask_by_name:
                match = mask_by_name[candidate_name]
                break
        if match is None:
            report.unmatched_raw.append(raw_path)
            expected = _candidate_mask_names(raw_path, (config.ground_truth_suffix,))[0]
            report.errors.append(
                f"No mask found for {raw_path.name}. Recommended mask name: {expected}"
            )
            continue

        matched_masks.add(match)
        report.pairs.append(VolumePair(volume_stem(raw_path), raw_path, match))

    report.unmatched_masks = [path for path in mask_files if path not in matched_masks]
    for path in report.unmatched_masks:
        report.warnings.append(f"Mask was not used because no raw volume matched it: {path.name}")

    if not raw_files:
        report.errors.append(f"No 3D image files found in raw_dir: {config.raw_dir}")
    if not mask_files:
        report.errors.append(f"No mask files found in ground_truth_dir: {config.ground_truth_dir}")

    return report


def inspect_custom_dataset(config: FinetuneConfig, base_dir: str | Path | None = None) -> DatasetInspectionReport:
    """Match files, validate shapes, and verify ground-truth masks."""
    paths = config.environment_paths(base_dir, verbose=False)
    config = config.resolve(paths["base"])
    report = find_volume_pairs(config, paths["base"])

    for pair in list(report.pairs):
        row = {
            "case_id": pair.case_id,
            "raw": pair.raw_path.name,
            "ground_truth": pair.mask_path.name,
        }
        try:
            raw = load_volume(pair.raw_path, channel_index=config.channel_index)
            mask = load_volume(pair.mask_path, channel_index=config.channel_index)
            row["raw_shape"] = tuple(int(x) for x in raw.shape)
            row["mask_shape"] = tuple(int(x) for x in mask.shape)

            if raw.shape != mask.shape:
                raise ValueError(f"shape mismatch: raw {raw.shape}, mask {mask.shape}")
            if float(np.max(raw)) == float(np.min(raw)):
                raise ValueError("raw volume is constant")

            mask_check = validate_binary_mask(mask, config.binary_threshold)
            row["mask_values"] = mask_check.unique_values
            row["foreground_fraction"] = f"{mask_check.foreground_fraction:.4%}"
            row["status"] = "ok"
            for warning in mask_check.warnings:
                report.warnings.append(f"{pair.mask_path.name}: {warning}")
        except Exception as exc:
            row["status"] = "error"
            row["problem"] = str(exc)
            report.errors.append(f"{pair.raw_path.name} / {pair.mask_path.name}: {exc}")
        report.rows.append(row)

    if len(report.pairs) == 1:
        report.warnings.append(
            "Only one labeled volume was found. Dataset preparation will partition that volume into non-overlapping train and validation subvolumes."
        )

    return report


def _write_dataset_json(dataset_dir: Path, num_training: int) -> None:
    dataset_json = {
        "channel_names": {"0": "image"},
        "labels": {"background": 0, "branch": 1},
        "numTraining": num_training,
        "file_ending": ".nii.gz",
    }
    (dataset_dir / "dataset.json").write_text(json.dumps(dataset_json, indent=2), encoding="utf-8")


def _write_case_mapping(dataset_dir: Path, case_mapping: dict[str, str]) -> None:
    (dataset_dir / "case_mapping.json").write_text(json.dumps(case_mapping, indent=2), encoding="utf-8")


def _load_case_mapping(dataset_dir: Path) -> dict[str, str]:
    mapping_path = dataset_dir / "case_mapping.json"
    if not mapping_path.exists():
        return {}
    return json.loads(mapping_path.read_text(encoding="utf-8"))


def _write_split_plan(
    dataset_dir: Path,
    train_ids: list[str],
    val_ids: list[str],
    mode: str,
    details: dict | None = None,
) -> None:
    plan = {
        "mode": mode,
        "train": train_ids,
        "val": val_ids,
    }
    if details:
        plan["details"] = details
    (dataset_dir / "split_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")


def _load_split_plan(dataset_dir: Path) -> tuple[list[str], list[str]] | None:
    plan_path = dataset_dir / "split_plan.json"
    if not plan_path.exists():
        return None
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    return list(plan.get("train", [])), list(plan.get("val", []))


def _dataset_signature(dataset_dir: Path) -> dict:
    signature = {"files": [], "metadata": {}}
    for subdir in ("imagesTr", "labelsTr"):
        for path in sorted((dataset_dir / subdir).glob("*.nii.gz")):
            signature["files"].append(
                {
                    "path": f"{subdir}/{path.name}",
                    "size": int(path.stat().st_size),
                }
            )
    for filename in ("dataset.json", "case_mapping.json", "split_plan.json"):
        path = dataset_dir / filename
        if path.exists():
            signature["metadata"][filename] = path.read_text(encoding="utf-8")
    return signature


def _write_dataset_signature(dataset_dir: Path) -> Path:
    signature_path = dataset_dir / "dataset_signature.json"
    signature_path.write_text(json.dumps(_dataset_signature(dataset_dir), indent=2), encoding="utf-8")
    return signature_path


def _current_dataset_signature(config: FinetuneConfig, base_dir: str | Path | None = None) -> dict | None:
    paths = config.environment_paths(base_dir, verbose=False)
    dataset_dir = Path(paths["nnUNet_raw"]) / dataset_folder_name(config.dataset_id, config.project_name)
    if not dataset_dir.exists():
        return None
    return _dataset_signature(dataset_dir)


def _saved_training_dataset_signature(config: FinetuneConfig, base_dir: str | Path | None = None) -> dict | None:
    signature_path = finetuned_output_dir(config, base_dir) / "dataset_signature.json"
    if not signature_path.exists():
        return None
    return json.loads(signature_path.read_text(encoding="utf-8"))


def _dataset_names_for_id(paths: dict[str, Path], dataset_id: int) -> set[str]:
    names: set[str] = set()
    pattern = re.compile(rf"^Dataset{int(dataset_id):03d}_.+")
    for root_key in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
        root = Path(paths[root_key])
        if not root.exists():
            continue
        for child in root.iterdir():
            if child.is_dir() and pattern.match(child.name):
                names.add(child.name)
    return names


def resolve_dataset_id_conflict(
    config: FinetuneConfig,
    base_dir: str | Path | None = None,
    max_lookahead: int = 200,
) -> tuple[FinetuneConfig, str | None]:
    """Pick a clean nnU-Net dataset id when the requested one is already used by another dataset name."""
    paths = config.environment_paths(base_dir, verbose=False)
    config = config.resolve(paths["base"])
    requested_id = int(config.dataset_id)
    requested_name = dataset_folder_name(requested_id, config.project_name)
    requested_existing_names = _dataset_names_for_id(paths, requested_id)

    for candidate_id in range(requested_id, requested_id + max_lookahead + 1):
        existing_names = _dataset_names_for_id(paths, candidate_id)
        candidate_name = dataset_folder_name(candidate_id, config.project_name)
        if not existing_names or existing_names == {candidate_name}:
            adjusted = config.resolve(paths["base"])
            adjusted.dataset_id = candidate_id
            if candidate_id == requested_id:
                return adjusted, None
            return (
                adjusted,
                (
                    f"Requested dataset id {requested_id} is already used by "
                    f"{', '.join(sorted(requested_existing_names or {requested_name}))}. "
                    f"Using dataset id {candidate_id} instead."
                ),
            )

    raise RuntimeError(
        f"Could not find a free nnU-Net dataset id starting at {requested_id}. "
        f"Tried {max_lookahead + 1} consecutive ids."
    )


def _resolve_split_ids(
    config: FinetuneConfig,
    dataset_dir: Path,
) -> tuple[list[str] | None, list[str] | None]:
    case_mapping = _load_case_mapping(dataset_dir)
    if not case_mapping:
        return None, None

    def convert(names: tuple[str, ...] | None, label: str) -> list[str] | None:
        if not names:
            return None
        resolved: list[str] = []
        missing: list[str] = []
        for name in names:
            case_id = case_mapping.get(name)
            if case_id is None:
                missing.append(name)
            else:
                resolved.append(case_id)
        if missing:
            raise ValueError(f"{label} case(s) not found in case mapping: {missing}")
        return resolved

    return convert(config.train_case_names, "Training"), convert(config.validation_case_names, "Validation")


def _automatic_split_ids(case_ids: list[str]) -> tuple[list[str], list[str], str]:
    if len(case_ids) <= 1:
        return case_ids, case_ids, "same_case_fallback"
    if len(case_ids) < 5:
        return case_ids[:-1], case_ids[-1:], "auto_single_validation"

    train_count = max(1, (len(case_ids) * 4) // 5)
    train_count = min(train_count, len(case_ids) - 1)
    return case_ids[:train_count], case_ids[train_count:], "auto_four_fifths"


def _partition_single_volume(
    raw: np.ndarray,
    mask: np.ndarray,
    divisors: tuple[int, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    if divisors is None:
        divisors = tuple(1 for _ in raw.shape)

    def partition_bounds(axis_length: int, min_partition_size: int) -> tuple[int, int, int] | None:
        if axis_length < max(8, min_partition_size * 2):
            return None
        gap = max(1, axis_length // 20)
        midpoint = axis_length // 2
        train_end = midpoint - gap // 2
        val_start = midpoint + (gap - gap // 2)
        if train_end < min_partition_size or (axis_length - val_start) < min_partition_size:
            gap = 0
            train_end = midpoint
            val_start = midpoint
        if train_end < min_partition_size or (axis_length - val_start) < min_partition_size:
            return None
        return gap, train_end, val_start

    selected_axis = None
    selected_bounds = None
    for axis in sorted(range(raw.ndim), key=lambda idx: raw.shape[idx], reverse=True):
        bounds = partition_bounds(int(raw.shape[axis]), int(divisors[axis]))
        if bounds is not None:
            selected_axis = axis
            selected_bounds = bounds
            break

    if selected_axis is None or selected_bounds is None:
        required = ", ".join(str(int(v)) for v in divisors)
        raise ValueError(
            f"could not partition single volume with shape {raw.shape} into train/validation subvolumes "
            f"that satisfy the minimum axis sizes required by the current plan: {required}"
        )

    split_axis = int(selected_axis)
    axis_length = int(raw.shape[split_axis])
    gap, train_end, val_start = selected_bounds

    train_selector = [slice(None)] * raw.ndim
    train_selector[split_axis] = slice(0, train_end)
    val_selector = [slice(None)] * raw.ndim
    val_selector[split_axis] = slice(val_start, axis_length)

    raw_train = raw[tuple(train_selector)]
    mask_train = mask[tuple(train_selector)]
    raw_val = raw[tuple(val_selector)]
    mask_val = mask[tuple(val_selector)]
    return raw_train, mask_train, raw_val, mask_val, {
        "axis": split_axis,
        "axis_length": axis_length,
        "gap_voxels": gap,
        "train_end": train_end,
        "val_start": val_start,
    }


def _load_reference_plans_dict(config: FinetuneConfig) -> dict | None:
    if config.reference_plans is None:
        return None
    source = Path(config.reference_plans)
    if not source.exists():
        return None
    return json.loads(source.read_text(encoding="utf-8"))


def _default_patch_size(config: FinetuneConfig) -> tuple[int, ...]:
    if config.reference_patch_size is not None:
        return tuple(int(v) for v in config.reference_patch_size)

    plans = _load_reference_plans_dict(config)
    if plans is not None:
        config_plans = plans.get("configurations", {}).get(config.configuration)
        if config_plans is not None and "patch_size" in config_plans:
            return tuple(int(v) for v in config_plans["patch_size"])
    return (128, 128, 64)


def _patch_divisors(config: FinetuneConfig, dims: int) -> tuple[int, ...]:
    plans = _load_reference_plans_dict(config)
    if plans is None:
        if dims == 3:
            return (32, 32, 16)
        return tuple(16 for _ in range(dims))

    config_plans = plans.get("configurations", {}).get(config.configuration)
    strides = (
        config_plans.get("architecture", {})
        .get("arch_kwargs", {})
        .get("strides")
        if config_plans is not None
        else None
    )
    if not strides:
        if dims == 3:
            return (32, 32, 16)
        return tuple(16 for _ in range(dims))

    divisors = [1] * dims
    for stride in strides:
        for axis, factor in enumerate(stride[:dims]):
            divisors[axis] *= int(factor)
    return tuple(max(1, value) for value in divisors)


def _select_source_split(
    config: FinetuneConfig,
    report: DatasetInspectionReport,
) -> tuple[list[str], list[str], str]:
    pair_ids = [pair.case_id for pair in report.pairs]
    if not pair_ids:
        return [], [], "no_cases"
    if config.train_case_names or config.validation_case_names:
        train_ids = list(config.train_case_names or [])
        val_ids = list(config.validation_case_names or [])
        if not train_ids or not val_ids:
            raise ValueError("When setting explicit train_case_names or validation_case_names, both must be set.")
        missing = [case_id for case_id in train_ids + val_ids if case_id not in pair_ids]
        if missing:
            raise ValueError(f"Explicit split refers to case(s) not found in raw/ground_truth: {missing}")
        return train_ids, val_ids, "explicit_case_names"
    if len(pair_ids) == 1:
        return pair_ids, pair_ids, "single_volume_partition"
    if len(pair_ids) < 5:
        return pair_ids[:-1], pair_ids[-1:], "auto_single_validation"
    train_count = max(1, (len(pair_ids) * 4) // 5)
    train_count = min(train_count, len(pair_ids) - 1)
    return pair_ids[:train_count], pair_ids[train_count:], "auto_four_fifths"


def _shape_tuple_stats(shapes: list[tuple[int, ...]]) -> tuple[tuple[int, ...] | None, tuple[int, ...] | None]:
    if not shapes:
        return None, None
    arr = np.asarray(shapes, dtype=np.int64)
    return tuple(int(v) for v in arr.min(axis=0)), tuple(int(v) for v in arr.max(axis=0))


def _shape_limited_patch_size(
    base_patch_size: tuple[int, ...],
    min_train_shape: tuple[int, ...],
    divisors: tuple[int, ...],
) -> tuple[int, ...]:
    limited: list[int] = []
    for axis, base in enumerate(base_patch_size):
        divisor = max(1, divisors[axis])
        axis_shape = int(min_train_shape[axis])
        if axis_shape >= divisor:
            capped = min(int(base), axis_shape)
            axis_patch = max(divisor, (capped // divisor) * divisor)
        else:
            axis_patch = divisor
        limited.append(int(axis_patch))
    return tuple(limited)


def _gpu_info() -> tuple[bool, str | None, float | None]:
    try:
        import torch
    except ImportError:
        return False, None, None

    if not torch.cuda.is_available():
        return False, None, None
    props = torch.cuda.get_device_properties(0)
    return True, props.name, props.total_memory / 1e9


def _vram_limited_patch_size(
    patch_size: tuple[int, ...],
    divisors: tuple[int, ...],
    gpu_vram_gb: float | None,
    baseline_vram_gb: float = 48.0,
) -> tuple[int, ...]:
    if gpu_vram_gb is None:
        return patch_size
    scale = min(1.0, max(0.25, gpu_vram_gb / baseline_vram_gb))
    target_voxels = int(np.prod(np.asarray(patch_size, dtype=np.int64)) * scale)
    adjusted = [int(v) for v in patch_size]
    while int(np.prod(np.asarray(adjusted, dtype=np.int64))) > target_voxels:
        reducible_axes = [axis for axis, size in enumerate(adjusted) if size - divisors[axis] >= divisors[axis]]
        if not reducible_axes:
            break
        axis = max(reducible_axes, key=lambda idx: adjusted[idx] / max(1, divisors[idx]))
        adjusted[axis] -= divisors[axis]
    return tuple(adjusted)


def preflight_training_setup(
    config: FinetuneConfig,
    base_dir: str | Path | None = None,
    report: DatasetInspectionReport | None = None,
    require_cuda: bool = True,
) -> TrainingPreflightReport:
    paths = config.environment_paths(base_dir, verbose=False)
    config = config.resolve(paths["base"])
    report = report or inspect_custom_dataset(config, paths["base"])

    errors = list(report.errors)
    warnings = list(report.warnings)
    notes: list[str] = []

    try:
        train_source_case_ids, validation_source_case_ids, split_mode = _select_source_split(config, report)
    except Exception as exc:
        train_source_case_ids, validation_source_case_ids, split_mode = [], [], "invalid_split"
        errors.append(str(exc))

    shape_by_case = {
        row["case_id"]: tuple(int(v) for v in row["raw_shape"])
        for row in report.rows
        if row.get("status") == "ok" and "raw_shape" in row
    }
    train_shapes: list[tuple[int, ...]] = []
    original_patch_size = _default_patch_size(config)
    divisors = _patch_divisors(config, len(original_patch_size))
    if split_mode == "single_volume_partition" and report.pairs:
        pair = report.pairs[0]
        try:
            raw = load_volume(pair.raw_path, channel_index=config.channel_index)
            mask = load_volume(pair.mask_path, channel_index=config.channel_index)
            mask_check = validate_binary_mask(mask, config.binary_threshold)
            raw_train, _, raw_val, _, partition_info = _partition_single_volume(
                raw,
                mask_check.mask,
                divisors=divisors,
            )
            train_shapes = [tuple(int(v) for v in raw_train.shape), tuple(int(v) for v in raw_val.shape)]
            notes.append(
                "One labeled volume was found, so the notebook will train on one partition and validate on a separate partition of the same volume."
            )
            notes.append(
                f"Single-volume partition axis: {partition_info['axis']} with a {partition_info['gap_voxels']}-voxel gap."
            )
        except Exception as exc:
            errors.append(f"Could not partition the single labeled volume for train/validation: {exc}")
    else:
        for case_id in train_source_case_ids:
            shape = shape_by_case.get(case_id)
            if shape is None:
                errors.append(f"Training case shape was not available for: {case_id}")
            else:
                train_shapes.append(shape)

    train_shape_min, train_shape_max = _shape_tuple_stats(train_shapes)
    if train_shape_min is not None:
        undersized_axes = [
            axis
            for axis in range(len(divisors))
            if int(train_shape_min[axis]) < int(divisors[axis])
        ]
        if undersized_axes:
            details = ", ".join(
                f"axis {axis}: shape {int(train_shape_min[axis])} < minimum {int(divisors[axis])}"
                for axis in undersized_axes
            )
            errors.append(
                "At least one training axis is too small for the current nnU-Net plan after splitting: "
                + details
            )
    recommended_patch_size = original_patch_size
    if train_shape_min is not None and not errors:
        recommended_patch_size = _shape_limited_patch_size(original_patch_size, train_shape_min, divisors)

    gpu_available, gpu_name, gpu_vram_gb = _gpu_info()
    if require_cuda and not gpu_available:
        errors.append("CUDA GPU not detected. This notebook expects a CUDA-capable GPU for 3D fine-tuning.")
    if gpu_available:
        before_vram_limit = recommended_patch_size
        recommended_patch_size = _vram_limited_patch_size(recommended_patch_size, divisors, gpu_vram_gb)
        if gpu_vram_gb is not None and gpu_vram_gb <= 24.5:
            warnings.append(
                f"Detected {gpu_vram_gb:.1f} GB VRAM. The patch size will be reduced when needed so 24 GB GPUs are less likely to run out of memory."
            )
        if recommended_patch_size != before_vram_limit:
            warnings.append(
                f"Patch size was reduced for available VRAM: {before_vram_limit} -> {recommended_patch_size}"
            )

    if train_shape_max is not None and any(train_shape_max[axis] > recommended_patch_size[axis] for axis in range(len(recommended_patch_size))):
        notes.append("Volumes larger than the patch size are fine. nnU-Net trains on patches sampled from each volume.")
    if train_shape_min is not None:
        smaller_axes = [
            axis
            for axis in range(len(recommended_patch_size))
            if train_shape_min[axis] < original_patch_size[axis]
        ]
        if smaller_axes:
            warnings.append(
                f"At least one training axis is smaller than the original patch size. The patch size will be adapted automatically: {original_patch_size} -> {recommended_patch_size}"
            )

    adjusted_config = config.resolve(paths["base"])
    adjusted_config.reference_patch_size = recommended_patch_size
    adjusted_config.reference_batch_size = 1 if config.reference_batch_size is None else min(int(config.reference_batch_size), 1)
    if not config.train_case_names and not config.validation_case_names:
        adjusted_config.train_case_names = None
        adjusted_config.validation_case_names = None

    return TrainingPreflightReport(
        config=adjusted_config,
        source_case_count=len(report.pairs),
        split_mode=split_mode,
        train_source_case_ids=train_source_case_ids,
        validation_source_case_ids=validation_source_case_ids,
        original_patch_size=tuple(int(v) for v in original_patch_size),
        recommended_patch_size=tuple(int(v) for v in recommended_patch_size),
        recommended_batch_size=int(adjusted_config.reference_batch_size or 1),
        train_shape_min=train_shape_min,
        train_shape_max=train_shape_max,
        gpu_available=gpu_available,
        gpu_name=gpu_name,
        gpu_vram_gb=gpu_vram_gb,
        notes=notes,
        warnings=warnings,
        errors=errors,
    )


def _write_split(
    paths: dict[str, Path],
    dataset_name: str,
    case_ids: list[str],
    train_ids: list[str] | None = None,
    val_ids: list[str] | None = None,
) -> None:
    split_dir = paths["nnUNet_preprocessed"] / dataset_name
    split_dir.mkdir(parents=True, exist_ok=True)
    if train_ids is not None or val_ids is not None:
        train_ids = train_ids or []
        val_ids = val_ids or []
    elif len(case_ids) == 1:
        train_ids = val_ids = case_ids
    else:
        train_ids, val_ids, _ = _automatic_split_ids(case_ids)
    splits = [{"train": train_ids, "val": val_ids}]
    (split_dir / "splits_final.json").write_text(json.dumps(splits, indent=2), encoding="utf-8")


def ensure_nnunet_split_file(config: FinetuneConfig, base_dir: str | Path | None = None) -> Path:
    """Rebuild splits_final.json from the prepared nnU-Net raw dataset."""
    paths = config.environment_paths(base_dir, verbose=False)
    config = config.resolve(paths["base"])
    dataset_name = dataset_folder_name(config.dataset_id, config.project_name)
    dataset_dir = paths["nnUNet_raw"] / dataset_name
    case_ids = [case_id for case_id, _, _ in _find_nnunet_training_pairs(dataset_dir)]
    if not case_ids:
        raise ValueError(f"No nnU-Net training cases found in {dataset_dir}")
    saved_split = _load_split_plan(dataset_dir)
    if saved_split is not None:
        train_ids, val_ids = saved_split
    else:
        train_ids, val_ids = _resolve_split_ids(config, dataset_dir)
    _write_split(paths, dataset_name, case_ids, train_ids=train_ids, val_ids=val_ids)
    return paths["nnUNet_preprocessed"] / dataset_name / "splits_final.json"


def prepare_nnunet_dataset(
    config: FinetuneConfig,
    base_dir: str | Path | None = None,
    report: DatasetInspectionReport | None = None,
) -> Path:
    """Convert validated raw/mask pairs into nnU-Net raw dataset format."""
    paths = config.environment_paths(base_dir, verbose=False)
    config = config.resolve(paths["base"])
    report = report or inspect_custom_dataset(config, paths["base"])
    report.raise_if_blocking_errors()

    dataset_name = dataset_folder_name(config.dataset_id, config.project_name)
    dataset_dir = paths["nnUNet_raw"] / dataset_name
    if dataset_dir.exists() and config.overwrite_dataset:
        shutil.rmtree(dataset_dir)

    images_tr = dataset_dir / "imagesTr"
    labels_tr = dataset_dir / "labelsTr"
    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)

    case_ids: list[str] = []
    case_mapping: dict[str, str] = {}
    split_mode = "explicit_case_names"
    split_details: dict | None = None
    train_ids: list[str] | None = None
    val_ids: list[str] | None = None
    divisors = _patch_divisors(config, len(_default_patch_size(config)))

    if len(report.pairs) == 1:
        pair = report.pairs[0]
        raw = load_volume(pair.raw_path, channel_index=config.channel_index)
        mask = load_volume(pair.mask_path, channel_index=config.channel_index)
        mask_check = validate_binary_mask(mask, config.binary_threshold)
        raw_train, mask_train, raw_val, mask_val, partition_info = _partition_single_volume(
            raw,
            mask_check.mask,
            divisors=divisors,
        )

        train_case_id = f"{_safe_case_id(pair.case_id)}_train_partition_001"
        val_case_id = f"{_safe_case_id(pair.case_id)}_val_partition_002"
        case_ids.extend([train_case_id, val_case_id])
        train_ids = [train_case_id]
        val_ids = [val_case_id]
        split_mode = "single_volume_partition"
        split_details = {"source_case": pair.case_id, **partition_info}

        save_nifti(raw_train, images_tr / f"{train_case_id}_0000.nii.gz", dtype=np.float32)
        save_nifti(mask_train, labels_tr / f"{train_case_id}.nii.gz", dtype=np.uint8)
        save_nifti(raw_val, images_tr / f"{val_case_id}_0000.nii.gz", dtype=np.float32)
        save_nifti(mask_val, labels_tr / f"{val_case_id}.nii.gz", dtype=np.uint8)
    else:
        for index, pair in enumerate(report.pairs, 1):
            case_id = f"{_safe_case_id(pair.case_id)}_{index:03d}"
            case_ids.append(case_id)
            case_mapping[pair.case_id] = case_id
            raw = load_volume(pair.raw_path, channel_index=config.channel_index)
            mask = load_volume(pair.mask_path, channel_index=config.channel_index)
            mask_check = validate_binary_mask(mask, config.binary_threshold)

            save_nifti(raw, images_tr / f"{case_id}_0000.nii.gz", dtype=np.float32)
            save_nifti(mask_check.mask, labels_tr / f"{case_id}.nii.gz", dtype=np.uint8)

        _write_case_mapping(dataset_dir, case_mapping)
        train_ids, val_ids = _resolve_split_ids(config, dataset_dir)
        if train_ids is None and val_ids is None:
            train_ids, val_ids, split_mode = _automatic_split_ids(case_ids)
        else:
            train_ids = train_ids or []
            val_ids = val_ids or []

    _write_dataset_json(dataset_dir, len(case_ids))
    if case_mapping and not (dataset_dir / "case_mapping.json").exists():
        _write_case_mapping(dataset_dir, case_mapping)
    _write_split_plan(dataset_dir, train_ids or [], val_ids or [], split_mode, details=split_details)
    _write_dataset_signature(dataset_dir)
    _write_split(paths, dataset_name, case_ids, train_ids=train_ids, val_ids=val_ids)

    print(f"Prepared nnU-Net dataset: {dataset_dir}")
    print(f"Training cases: {len(case_ids)}")
    print(f"Split: {len(train_ids or [])} train / {len(val_ids or [])} val ({split_mode})")
    return dataset_dir


def default_pretrained_weights(config: FinetuneConfig, base_dir: str | Path | None = None) -> Path:
    paths = config.environment_paths(base_dir, verbose=False)
    return (
        paths["nnUNet_results"]
        / config.pretrained_dataset_name
        / config.pretrained_trainer_dir
        / f"fold_{config.pretrained_fold}"
        / "checkpoint_best.pth"
    )


def ensure_pretrained_weights(config: FinetuneConfig, base_dir: str | Path | None = None) -> Path:
    """Return a usable pretrained checkpoint, downloading the default one when needed."""
    paths = config.environment_paths(base_dir, verbose=False)
    config = config.resolve(paths["base"])

    if config.pretrained_weights is not None:
        checkpoint = Path(config.pretrained_weights)
        if not checkpoint.exists():
            raise FileNotFoundError(f"Configured pretrained checkpoint not found: {checkpoint}")
        return checkpoint

    checkpoint = default_pretrained_weights(config, paths["base"])
    if checkpoint.exists():
        return checkpoint
    return download_and_install_pretrained_weights(paths)


def default_vessel12_reference_plans(config: FinetuneConfig, base_dir: str | Path | None = None) -> Path:
    """Return a pretrained-compatible VESSEL12 plans file for demo fine-tuning."""
    paths = config.environment_paths(base_dir, verbose=False)
    candidates = [
        paths["weights"] / "DeepBranchAI_Zenodo" / "configs" / "DeepBranchAI_VESSEL12_nnUNetPlans.json",
        paths["nnUNet_preprocessed"] / "Dataset3005_Mitochondria" / "nnUNetPlans.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return download_vessel12_reference_plans(paths)


def install_reference_plans(config: FinetuneConfig, base_dir: str | Path | None = None) -> Path:
    """Copy a compatible plans file into this dataset's nnU-Net preprocessed folder."""
    config = config.resolve(base_dir)
    if config.reference_plans is None:
        raise ValueError("config.reference_plans must be set before installing reference plans.")

    source = Path(config.reference_plans)
    if not source.exists():
        raise FileNotFoundError(f"Reference plans not found: {source}")

    target_dir = Path(config.nnunet_preprocessed) / dataset_folder_name(config.dataset_id, config.project_name)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{config.plans}.json"
    plans = json.loads(source.read_text(encoding="utf-8"))
    plans["dataset_name"] = dataset_folder_name(config.dataset_id, config.project_name)
    config_plans = plans.get("configurations", {}).get(config.configuration)
    if config_plans is None:
        raise KeyError(f"Configuration {config.configuration!r} not found in reference plans: {source}")
    if config.reference_patch_size is not None:
        config_plans["patch_size"] = [int(v) for v in config.reference_patch_size]
    if config.reference_batch_size is not None:
        config_plans["batch_size"] = int(config.reference_batch_size)
    target.write_text(json.dumps(plans, indent=2), encoding="utf-8")
    raw_dataset_json = Path(config.nnunet_raw) / dataset_folder_name(config.dataset_id, config.project_name) / "dataset.json"
    preprocessed_dataset_json = target_dir / "dataset.json"
    if raw_dataset_json.exists():
        shutil.copy2(str(raw_dataset_json), str(preprocessed_dataset_json))
    print(f"Reference plans ready: {target}")
    return target


def clear_preprocessed_dataset(config: FinetuneConfig, base_dir: str | Path | None = None) -> Path:
    """Remove this dataset's nnU-Net preprocessed folder so preprocessing can start clean."""
    config = config.resolve(base_dir)
    target = Path(config.nnunet_preprocessed) / dataset_folder_name(config.dataset_id, config.project_name)
    if target.exists():
        shutil.rmtree(target)
    return target


def finetuned_output_dir(config: FinetuneConfig, base_dir: str | Path | None = None) -> Path:
    """Return the nnU-Net output directory for this fine-tuning run."""
    paths = config.environment_paths(base_dir, verbose=False)
    dataset_name = dataset_folder_name(config.dataset_id, config.project_name)
    trainer_dir = f"{config.trainer}__{config.plans}__{config.configuration}"
    return paths["nnUNet_results"] / dataset_name / trainer_dir / f"fold_{config.fold}"


def finetuned_checkpoint_path(config: FinetuneConfig, base_dir: str | Path | None = None) -> Path:
    """Return the expected checkpoint path for this fine-tuning run."""
    return finetuned_output_dir(config, base_dir) / "checkpoint_best.pth"


def training_graph_path(config: FinetuneConfig, base_dir: str | Path | None = None) -> Path:
    """Return nnU-Net's training progress figure for this fine-tuning run."""
    return finetuned_output_dir(config, base_dir) / "progress.png"


def _load_existing_training_plans(config: FinetuneConfig, base_dir: str | Path | None = None) -> dict | None:
    config = config.resolve(base_dir)
    candidates = [
        finetuned_output_dir(config, base_dir).parent / "plans.json",
        Path(config.nnunet_preprocessed) / dataset_folder_name(config.dataset_id, config.project_name) / f"{config.plans}.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return json.loads(candidate.read_text(encoding="utf-8"))
    return None


def _training_plan_matches_config(config: FinetuneConfig, base_dir: str | Path | None = None) -> bool:
    plans = _load_existing_training_plans(config, base_dir)
    if plans is None:
        return False

    config_plans = plans.get("configurations", {}).get(config.configuration)
    if config_plans is None:
        return False

    desired_patch = tuple(int(v) for v in _default_patch_size(config))
    existing_patch = tuple(int(v) for v in config_plans.get("patch_size", []))
    if existing_patch != desired_patch:
        return False

    desired_batch = int(config.reference_batch_size) if config.reference_batch_size is not None else int(
        config_plans.get("batch_size", 1)
    )
    existing_batch = int(config_plans.get("batch_size", 1))
    return existing_batch == desired_batch


def completed_training_epochs(config: FinetuneConfig, base_dir: str | Path | None = None) -> int | None:
    """Return the number of completed epochs recorded in the latest nnU-Net training log."""
    output_dir = finetuned_output_dir(config, base_dir)
    logs = sorted(output_dir.glob("training_log_*.txt"))
    if not logs:
        return None

    pattern = re.compile(r"Epoch\s+(\d+)")
    epoch_numbers: list[int] = []
    text = logs[-1].read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        match = pattern.search(line)
        if match is not None:
            epoch_numbers.append(int(match.group(1)))
    if not epoch_numbers:
        return None
    return max(epoch_numbers) + 1


def clear_finetuned_output(config: FinetuneConfig, base_dir: str | Path | None = None) -> Path:
    """Remove this run's nnU-Net output folder so fine-tuning can restart cleanly."""
    target = finetuned_output_dir(config, base_dir)
    if target.exists():
        shutil.rmtree(target)
    return target


def show_training_graphs(
    config: FinetuneConfig,
    base_dir: str | Path | None = None,
    dpi: int = 200,
) -> Path:
    """Display nnU-Net's training progress figure and return its path."""
    graph_path = training_graph_path(config, base_dir)
    if not graph_path.exists():
        raise FileNotFoundError(f"Training graph not found: {graph_path}")

    image = mpimg.imread(graph_path)
    plt.figure(figsize=(12, 8), dpi=dpi)
    plt.imshow(image)
    plt.axis("off")
    plt.tight_layout()
    render_figure(plt.gcf())
    return graph_path


def finetune_model(config: FinetuneConfig, base_dir: str | Path | None = None) -> Path:
    """Run preprocessing and fine-tune in one call."""
    paths = config.environment_paths(base_dir, verbose=False)
    config = config.resolve(paths["base"])
    checkpoint = finetuned_checkpoint_path(config, paths["base"])
    if checkpoint.exists():
        completed_epochs = completed_training_epochs(config, paths["base"])
        plan_matches = _training_plan_matches_config(config, paths["base"])
        current_signature = _current_dataset_signature(config, paths["base"])
        saved_signature = _saved_training_dataset_signature(config, paths["base"])
        dataset_matches = current_signature is not None and saved_signature == current_signature
        if (
            config.reuse_existing_training
            and completed_epochs is not None
            and completed_epochs >= config.max_epochs
            and plan_matches
            and dataset_matches
        ):
            print(f"Fine-tuning already complete: {checkpoint} ({completed_epochs} epochs)")
            return checkpoint

        if config.reuse_existing_training:
            reasons: list[str] = []
            if completed_epochs is None or completed_epochs < config.max_epochs:
                reasons.append(
                    f"found {completed_epochs if completed_epochs is not None else 'unknown'} epoch(s), requested {config.max_epochs}"
                )
            if not plan_matches:
                reasons.append("the saved training plan does not match the requested patch or batch settings")
            if not dataset_matches:
                reasons.append("the prepared dataset does not match the checkpoint that is already on disk")
            print("Existing fine-tuning output will be replaced because " + " and ".join(reasons) + ".")
        else:
            print("reuse_existing_training is disabled. Re-running training from the pretrained checkpoint.")
        clear_finetuned_output(config, paths["base"])
    if config.run_preprocessing:
        if config.reference_plans is None:
            run_planning_and_preprocessing(
                config.dataset_id,
                verify_dataset_integrity=True,
                num_processes=config.num_processes,
            )
        else:
            clear_preprocessed_dataset(config, paths["base"])
            extract_fingerprints_only(
                config.dataset_id,
                verify_dataset_integrity=True,
                num_processes=config.num_processes,
            )
            install_reference_plans(config, paths["base"])
            ensure_nnunet_split_file(config, paths["base"])
            preprocess_with_existing_plans(
                config.dataset_id,
                plans_identifier=config.plans,
                configurations=config.preprocess_configurations,
                num_processes=config.num_processes,
            )

    pretrained = ensure_pretrained_weights(config, paths["base"])
    checkpoint = finetune_from_pretrained(
        dataset_id=config.dataset_id,
        fold=config.fold,
        pretrained_weights=pretrained,
        trainer=config.trainer,
        plans=config.plans,
        configuration=config.configuration,
        max_epochs=config.max_epochs,
        num_processes=config.num_processes,
        setup_env=False,
    )
    current_signature = _current_dataset_signature(config, paths["base"])
    if current_signature is not None:
        output_dir = finetuned_output_dir(config, paths["base"])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "dataset_signature.json").write_text(json.dumps(current_signature, indent=2), encoding="utf-8")
    return checkpoint


def prepare_prediction_inputs(config: FinetuneConfig, base_dir: str | Path | None = None) -> Path:
    """Convert files in predict_dir to nnU-Net prediction input format."""
    paths = config.environment_paths(base_dir, verbose=False)
    config = config.resolve(paths["base"])
    files = list_volume_files(config.predict_dir)
    if not files:
        raise ValueError(f"No prediction volumes found in predict_dir: {config.predict_dir}")

    input_dir = paths["tmp"] / "custom_finetune_predict_input"
    if input_dir.exists():
        shutil.rmtree(input_dir)
    input_dir.mkdir(parents=True, exist_ok=True)

    for index, path in enumerate(files, 1):
        case_id = f"{_safe_case_id(volume_stem(path))}_{index:03d}"
        raw = load_volume(path, channel_index=config.channel_index)
        save_nifti(raw, input_dir / f"{case_id}_0000.nii.gz", dtype=np.float32)

    print(f"Prepared {len(files)} prediction volume(s): {input_dir}")
    return input_dir


def predict_with_finetuned_model(config: FinetuneConfig, base_dir: str | Path | None = None) -> Path:
    """Run the finetuned model on files in predict_dir."""
    paths = config.environment_paths(base_dir, verbose=False)
    config = config.resolve(paths["base"])
    input_dir = prepare_prediction_inputs(config, paths["base"])
    output_dir = Path(config.output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predict_nnunet(
        input_dir=input_dir,
        output_dir=output_dir,
        dataset_id=config.dataset_id,
        fold=config.fold,
        trainer=config.trainer,
        configuration=config.configuration,
    )

    for nii_path in sorted(output_dir.glob("*.nii.gz")):
        arr = np.asanyarray(nib.load(str(nii_path)).dataobj)
        save_tiff(arr.astype(np.uint8), output_dir / f"{nii_path.name[:-7]}.tif")

    print(f"Predictions saved to: {output_dir}")
    return output_dir


def show_folder_layout(config: FinetuneConfig, base_dir: str | Path | None = None) -> None:
    """Print the expected folder layout."""
    config = create_custom_finetune_folders(config, base_dir)
    print("Put training volumes here:")
    print(f"  raw:          {config.raw_dir}")
    print(f"  ground_truth: {config.ground_truth_dir}")
    print()
    print("Use matching names:")
    print("  raw/sample01.tif")
    print(f"  ground_truth/sample01{config.ground_truth_suffix}.tif")
    print()
    print("Put volumes to segment after fine-tuning here:")
    print(f"  predict:      {config.predict_dir}")

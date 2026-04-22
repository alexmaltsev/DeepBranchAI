"""Helpers for the VESSEL12 inference demo notebook."""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import tifffile

from .downloads import (
    download_and_install_vessel12_weights,
    download_vessel12_demo_data,
)
from .image_io import save_nifti, save_tiff
from .nnunet_runner import predict_nnunet
from .paths import setup_environment
from .visualization import render_figure


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


def _volume_stem(path: str | Path) -> str:
    path = Path(path)
    if path.name.lower().endswith(".nii.gz"):
        return path.name[:-7]
    return path.stem


def _read_volume_any(path: str | Path) -> np.ndarray:
    path = Path(path)
    name = path.name.lower()
    if name.endswith((".tif", ".tiff")):
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
        raise ValueError(f"Unsupported VESSEL12 input file: {path}")
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
    for axis in (arr.ndim - 1, 0, 1):
        if axis in candidates:
            return axis
    return candidates[0]


def _sanitize_volume(arr: np.ndarray, path: Path | None = None) -> tuple[np.ndarray, list[str]]:
    warnings: list[str] = []
    name = path.name if path is not None else "input"
    if arr.ndim == 3:
        return arr.astype(np.float32), warnings
    if arr.ndim != 4:
        raise ValueError(f"{name} must be a 3D volume or 4D with a channel axis; got shape {arr.shape}")

    channel_axis = _find_channel_axis(arr)
    if channel_axis is None:
        raise ValueError(f"{name} has shape {arr.shape}, but the channel axis could not be inferred.")
    channels = np.moveaxis(arr, channel_axis, -1)
    n_channels = channels.shape[-1]
    if n_channels == 1:
        warnings.append(f"{name}: squeezed single-channel axis {channel_axis}.")
        return np.asarray(channels[..., 0], dtype=np.float32), warnings
    if n_channels in (3, 4):
        rgb = channels[..., :3].astype(np.float32)
        gray = (0.2989 * rgb[..., 0]) + (0.5870 * rgb[..., 1]) + (0.1140 * rgb[..., 2])
        warnings.append(f"{name}: converted {n_channels}-channel input to grayscale.")
        return gray.astype(np.float32), warnings
    raise ValueError(f"{name} has {n_channels} channels. Provide a single-channel grayscale volume.")


def _find_demo_raw_volume(data_dir: Path, case_name: str) -> Path | None:
    preferred = [
        f"{case_name}_raw.tif",
        f"{case_name}_raw.tiff",
        f"{case_name}.nii.gz",
        f"{case_name}.nii",
        f"{case_name}.mha",
        f"{case_name}.mhd",
    ]
    for name in preferred:
        for path in data_dir.rglob(name):
            return path
    return None


def _find_demo_annotation(data_dir: Path, case_name: str) -> Path | None:
    for name in (f"{case_name}_Annotations.csv", f"{case_name}.csv"):
        for path in data_dir.rglob(name):
            return path
    return None


def _annotation_coordinates(
    df: pd.DataFrame,
    mask: np.ndarray | None,
    volume_shape: tuple[int, ...],
) -> tuple[np.ndarray, str]:
    coords = df[["x", "y", "z"]].to_numpy(dtype=np.float64)
    depth, height, width = volume_shape
    candidates: list[tuple[np.ndarray, str]] = []
    for scale, label in ((1.0, "original coordinates"), (2.0, "2x scaled coordinates")):
        scaled = np.rint(coords * scale).astype(np.int64)
        if (
            scaled[:, 0].min() >= 0
            and scaled[:, 1].min() >= 0
            and scaled[:, 2].min() >= 0
            and scaled[:, 0].max() < width
            and scaled[:, 1].max() < height
            and scaled[:, 2].max() < depth
        ):
            candidates.append((scaled, label))
    if not candidates:
        raise ValueError(
            f"Annotation coordinates do not fit the prediction volume with either original or 2x-scaled coordinates. "
            f"Volume shape is {volume_shape}."
        )
    if len(candidates) == 1 or mask is None:
        return candidates[0]

    true_labels = df["label"].to_numpy(dtype=np.int64)
    best_score = None
    best_candidate = candidates[0]
    for scaled, label in candidates:
        pred_labels = mask[scaled[:, 2], scaled[:, 1], scaled[:, 0]]
        score = float((pred_labels == true_labels).mean())
        if best_score is None or score > best_score:
            best_score = score
            best_candidate = (scaled, label)
    return best_candidate


@dataclass
class Vessel12DemoConfig:
    base_dir: Path | str | None = None
    storage_dir: Path | str | None = None
    data_dir: Path | str | None = None
    weights_dir: Path | str | None = None
    tmp_dir: Path | str | None = None
    nnunet_raw: Path | str | None = None
    nnunet_preprocessed: Path | str | None = None
    nnunet_results: Path | str | None = None
    input_dir: Path | str = Path("vessel12_demo/input")
    annotation_dir: Path | str = Path("vessel12_demo/annotations")
    output_dir: Path | str = Path("vessel12_demo/predictions")
    case_name: str = "VESSEL12_21"
    threshold: float = 0.5
    dataset_id: int = 3005
    fold: int = 2
    trainer: str = "nnUNetTrainer_100epochs"
    configuration: str = "3d_fullres"

    def environment_paths(self, base_dir: str | Path | None = None, verbose: bool = False) -> dict[str, Path]:
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

    def resolve(self, base_dir: str | Path | None = None) -> "Vessel12DemoConfig":
        paths = self.environment_paths(base_dir, verbose=False)
        resolved = Vessel12DemoConfig(**self.__dict__)
        resolved.base_dir = paths["base"]
        resolved.storage_dir = paths["storage"]
        resolved.data_dir = paths["data"]
        resolved.weights_dir = paths["weights"]
        resolved.tmp_dir = paths["tmp"]
        resolved.nnunet_raw = paths["nnUNet_raw"]
        resolved.nnunet_preprocessed = paths["nnUNet_preprocessed"]
        resolved.nnunet_results = paths["nnUNet_results"]
        resolved.input_dir = _resolve_under_root(resolved.input_dir, paths["data"], "data")
        resolved.annotation_dir = _resolve_under_root(resolved.annotation_dir, paths["data"], "data")
        resolved.output_dir = _resolve_under_root(resolved.output_dir, paths["data"], "data")
        return resolved


@dataclass
class Vessel12DemoState:
    config: Vessel12DemoConfig
    used_demo_data: bool
    input_path: Path
    annotation_path: Path | None
    output_dir: Path
    volume: np.ndarray
    warnings: list[str] = field(default_factory=list)
    probability_map: np.ndarray | None = None
    mask: np.ndarray | None = None
    annotation_table: pd.DataFrame | None = None
    annotation_scale_note: str | None = None
    validation_metrics: dict[str, float] | None = None

    def show(self) -> None:
        print(f"Workspace root:         {self.config.storage_dir}")
        print(f"Managed input volume:   {self.input_path}")
        print(f"Annotation CSV:         {self.annotation_path if self.annotation_path else 'none'}")
        print(f"Segmentation output:    {self.output_dir}")
        print(f"Demo mode:              {self.used_demo_data}")
        print(f"Volume shape:           {self.volume.shape}")
        if self.warnings:
            print("\nWarnings:")
            for message in self.warnings:
                print(f"  - {message}")


def prepare_vessel12_demo(
    config: Vessel12DemoConfig,
    inference_input_path: str | Path | None = None,
    annotation_csv_path: str | Path | None = None,
    base_dir: str | Path | None = None,
    reset_workspace: bool = True,
) -> Vessel12DemoState:
    paths = config.environment_paths(base_dir, verbose=False)
    config = config.resolve(paths["base"])
    download_and_install_vessel12_weights(paths)

    input_dir = Path(config.input_dir)
    annotation_dir = Path(config.annotation_dir)
    output_dir = Path(config.output_dir)
    for folder in (input_dir, annotation_dir, output_dir):
        folder.mkdir(parents=True, exist_ok=True)
        if reset_workspace:
            for child in folder.iterdir():
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()

    warnings: list[str] = []
    used_demo_data = inference_input_path is None
    if used_demo_data:
        demo_dir = download_vessel12_demo_data(paths)
        source_volume = _find_demo_raw_volume(demo_dir, config.case_name)
        if source_volume is None:
            raise FileNotFoundError(f"{config.case_name} raw volume was not found in downloaded demo data.")
        if annotation_csv_path is None:
            annotation_csv_path = _find_demo_annotation(demo_dir, config.case_name)
    else:
        source_volume = Path(inference_input_path)
        if not source_volume.exists():
            raise FileNotFoundError(f"Inference input volume not found: {source_volume}")

    volume, volume_warnings = _sanitize_volume(_read_volume_any(source_volume), source_volume)
    warnings.extend(volume_warnings)
    staged_input = input_dir / f"{_volume_stem(source_volume)}.tif"
    save_tiff(volume.astype(np.float32), staged_input)

    staged_annotation = None
    if annotation_csv_path is not None:
        annotation_csv_path = Path(annotation_csv_path)
        if not annotation_csv_path.exists():
            raise FileNotFoundError(f"Annotation CSV not found: {annotation_csv_path}")
        staged_annotation = annotation_dir / annotation_csv_path.name
        shutil.copy2(str(annotation_csv_path), str(staged_annotation))

    return Vessel12DemoState(
        config=config,
        used_demo_data=used_demo_data,
        input_path=staged_input,
        annotation_path=staged_annotation,
        output_dir=output_dir,
        volume=volume,
        warnings=warnings,
    )


def run_vessel12_demo(state: Vessel12DemoState) -> Path:
    input_tmp = Path(state.config.tmp_dir) / "vessel12_demo_inference_input"
    if input_tmp.exists():
        shutil.rmtree(input_tmp)
    input_tmp.mkdir(parents=True, exist_ok=True)

    case_id = f"{_volume_stem(state.input_path)}_0000"
    input_nii = input_tmp / f"{case_id}.nii.gz"
    save_nifti(state.volume.astype(np.float32), input_nii)

    output_dir = Path(state.output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predict_nnunet(
        input_dir=input_tmp,
        output_dir=output_dir,
        dataset_id=state.config.dataset_id,
        fold=state.config.fold,
        trainer=state.config.trainer,
        configuration=state.config.configuration,
        save_probabilities=True,
    )

    prob_map, mask = load_vessel12_prediction(output_dir, state.volume.shape, threshold=state.config.threshold)
    save_tiff(mask.astype(np.uint8), output_dir / f"{_volume_stem(state.input_path)}_segmentation.tif")
    state.probability_map = prob_map
    state.mask = mask
    return output_dir


def load_vessel12_prediction(
    output_dir: str | Path,
    expected_shape: tuple[int, ...],
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    output_dir = Path(output_dir)
    npz_files = sorted(output_dir.glob("*.npz"))
    nii_files = sorted(output_dir.glob("*.nii.gz"))
    if npz_files:
        data = np.load(str(npz_files[0]))
        arr = data[data.files[0]]
        prob_map = arr[1] if arr.ndim == 4 else arr.astype(np.float32)
    elif nii_files:
        prob_map = np.asanyarray(nib.load(str(nii_files[0])).dataobj).astype(np.float32)
    else:
        raise FileNotFoundError(f"No prediction files were found in {output_dir}")

    prob_map = np.asarray(prob_map, dtype=np.float32)
    if prob_map.shape != expected_shape and prob_map.ndim == 3 and prob_map.transpose((2, 1, 0)).shape == expected_shape:
        prob_map = prob_map.transpose((2, 1, 0))
    if prob_map.shape != expected_shape:
        raise ValueError(f"Prediction shape {prob_map.shape} does not match the input volume shape {expected_shape}.")

    mask = (prob_map >= threshold).astype(np.uint8)
    return prob_map, mask


def compute_annotation_metrics(state: Vessel12DemoState) -> dict[str, float]:
    if state.mask is None:
        raise ValueError("Run run_vessel12_demo(...) before computing validation metrics.")
    if state.annotation_path is None:
        raise ValueError("No annotation CSV was provided for validation.")

    df = pd.read_csv(str(state.annotation_path), header=None, names=["x", "y", "z", "label"])
    coords, scale_note = _annotation_coordinates(df, state.mask, state.mask.shape)
    xs, ys, zs = coords[:, 0], coords[:, 1], coords[:, 2]
    true_labels = df["label"].to_numpy(dtype=np.int64)
    pred_labels = state.mask[zs, ys, xs]

    n_points = int(len(true_labels))
    n_match = int((pred_labels == true_labels).sum())
    n_vessel = int((true_labels == 1).sum())
    n_non_vessel = int((true_labels == 0).sum())
    n_vessel_match = int((pred_labels[true_labels == 1] == 1).sum()) if n_vessel else 0
    n_non_vessel_match = int((pred_labels[true_labels == 0] == 0).sum()) if n_non_vessel else 0

    metrics = {
        "points": float(n_points),
        "correct": float(n_match),
        "accuracy": float(n_match / n_points) if n_points else 0.0,
        "vessel_sensitivity": float(n_vessel_match / n_vessel) if n_vessel else 0.0,
        "non_vessel_specificity": float(n_non_vessel_match / n_non_vessel) if n_non_vessel else 0.0,
    }
    state.annotation_table = df.assign(x=coords[:, 0], y=coords[:, 1], z=coords[:, 2])
    state.annotation_scale_note = scale_note
    state.validation_metrics = metrics
    return metrics


def show_vessel12_segmentation(state: Vessel12DemoState, dpi: int = 200) -> None:
    if state.mask is None:
        raise ValueError("Run run_vessel12_demo(...) before showing results.")
    mid = state.volume.shape[0] // 2
    raw_slice = state.volume[mid]
    mask_slice = state.mask[mid]
    rgb = np.stack([raw_slice] * 3, axis=-1).astype(np.float32)
    rgb = rgb / max(1.0, float(rgb.max()))
    vessel = mask_slice > 0
    rgb[vessel, 0] = 1.0
    rgb[vessel, 1] *= 0.3
    rgb[vessel, 2] *= 0.3

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=dpi)
    axes[0].imshow(raw_slice, cmap="gray")
    axes[0].set_title(f"Raw volume (z={mid})")
    axes[1].imshow(mask_slice, cmap="Reds", vmin=0, vmax=1)
    axes[1].set_title("Segmented vessels")
    axes[2].imshow(rgb)
    axes[2].set_title("Overlay")
    for axis in axes:
        axis.axis("off")
    plt.tight_layout()
    render_figure(fig)


def show_vessel12_probability(state: Vessel12DemoState, dpi: int = 200) -> None:
    if state.probability_map is None:
        raise ValueError("Run run_vessel12_demo(...) before showing probabilities.")
    mid = state.volume.shape[0] // 2
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=dpi)
    axes[0].imshow(state.volume[mid], cmap="gray")
    axes[0].set_title(f"Raw volume (z={mid})")
    axes[1].imshow(state.probability_map[mid], cmap="hot", vmin=0, vmax=1)
    axes[1].set_title("Vessel probability")
    for axis in axes:
        axis.axis("off")
    plt.tight_layout()
    render_figure(fig)


def show_vessel12_validation(state: Vessel12DemoState, dpi: int = 200) -> None:
    if state.annotation_table is None or state.mask is None:
        raise ValueError("Run compute_annotation_metrics(...) before showing validation overlays.")

    ann_slices = sorted(state.annotation_table["z"].unique())
    ann_mid = int(ann_slices[len(ann_slices) // 2])
    df_slice = state.annotation_table[state.annotation_table["z"] == ann_mid]

    fig, axes = plt.subplots(1, 2, figsize=(16, 8), dpi=dpi)
    axes[0].imshow(state.volume[ann_mid], cmap="gray")
    for _, row in df_slice.iterrows():
        color = "lime" if int(row["label"]) == 1 else "cyan"
        axes[0].plot(row["x"], row["y"], "o", color=color, markersize=6, markeredgecolor="black", markeredgewidth=0.5)
    axes[0].set_title(f"Annotation points (z={ann_mid})")
    axes[0].axis("off")

    rgb = np.stack([state.volume[ann_mid]] * 3, axis=-1).astype(np.float32)
    rgb = rgb / max(1.0, float(rgb.max()))
    vessel = state.mask[ann_mid] > 0
    rgb[vessel, 0] = 1.0
    rgb[vessel, 1] *= 0.3
    rgb[vessel, 2] *= 0.3
    axes[1].imshow(rgb)
    for _, row in df_slice.iterrows():
        correct = int(state.mask[int(row["z"]), int(row["y"]), int(row["x"])]) == int(row["label"])
        color = "lime" if correct else "red"
        marker = "o" if correct else "X"
        axes[1].plot(row["x"], row["y"], marker, color=color, markersize=8, markeredgecolor="black", markeredgewidth=0.5)
    axes[1].set_title("Validation overlay")
    axes[1].axis("off")
    plt.tight_layout()
    render_figure(fig)

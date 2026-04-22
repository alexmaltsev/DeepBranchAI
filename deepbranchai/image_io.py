"""Volume I/O and mask validation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import tifffile


SUPPORTED_EXTENSIONS = (".tif", ".tiff", ".nii", ".nii.gz", ".mha", ".mhd")


@dataclass(frozen=True)
class MaskCheck:
    mask: np.ndarray
    unique_values: list[float]
    foreground_fraction: float
    warnings: list[str]


def is_volume_file(path: str | Path) -> bool:
    name = Path(path).name.lower()
    return any(name.endswith(ext) for ext in SUPPORTED_EXTENSIONS)


def volume_stem(path: str | Path) -> str:
    path = Path(path)
    name = path.name
    if name.lower().endswith(".nii.gz"):
        return name[:-7]
    return path.stem


def list_volume_files(directory: str | Path) -> list[Path]:
    directory = Path(directory)
    if not directory.exists():
        return []
    return sorted(path for path in directory.rglob("*") if path.is_file() and is_volume_file(path))


def load_volume(path: str | Path, channel_index: int | None = None) -> np.ndarray:
    """Load a 3D volume from TIFF, NIfTI, MHA, or MHD."""
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
        raise ValueError(f"Unsupported volume file: {path}")

    arr = np.asarray(arr)
    if arr.ndim == 4:
        if channel_index is None:
            raise ValueError(
                f"{path.name} appears to be multi-channel with shape {arr.shape}. "
                "Set channel_index in FinetuneConfig."
            )
        if arr.shape[-1] <= 8:
            arr = arr[..., channel_index]
        elif arr.shape[1] <= 8:
            arr = arr[:, channel_index, ...]
        else:
            raise ValueError(
                f"{path.name} is 4D with shape {arr.shape}, but the channel axis is ambiguous."
            )

    if arr.ndim != 3:
        raise ValueError(f"{path.name} must be a 3D volume after channel selection; got shape {arr.shape}")

    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{path.name} contains NaN or infinite values")

    return arr


def validate_binary_mask(mask: np.ndarray, binary_threshold: float | None = None) -> MaskCheck:
    """Validate and return a uint8 0/1 mask."""
    warnings: list[str] = []
    if not np.all(np.isfinite(mask)):
        raise ValueError("mask contains NaN or infinite values")

    unique = np.unique(mask)
    unique_preview = [float(x) for x in unique[:12]]

    if binary_threshold is not None:
        out = (mask > binary_threshold).astype(np.uint8)
        warnings.append(f"Mask was binarized with threshold > {binary_threshold}.")
    elif unique.size == 1:
        raise ValueError(f"mask has one value only: {unique_preview}")
    elif unique.size == 2:
        low, high = unique
        out = (mask == high).astype(np.uint8)
        if not ({float(low), float(high)} <= {0.0, 1.0} or {float(low), float(high)} <= {0.0, 255.0}):
            warnings.append(f"Mask has two values {unique_preview}; using the larger value as foreground.")
    else:
        raise ValueError(
            "mask is not clearly binary. "
            f"Found {unique.size} values; first values are {unique_preview}. "
            "Set binary_threshold if this mask should be thresholded."
        )

    foreground_fraction = float(out.mean())
    if foreground_fraction <= 0:
        raise ValueError("mask has no foreground voxels")
    if foreground_fraction >= 0.95:
        raise ValueError(f"mask is {foreground_fraction:.1%} foreground, which is probably not a valid mask")
    if foreground_fraction < 0.0001:
        warnings.append(f"Mask foreground is very sparse ({foreground_fraction:.4%}).")

    return MaskCheck(out, unique_preview, foreground_fraction, warnings)


def save_nifti(volume: np.ndarray, path: str | Path, dtype: np.dtype | type = np.float32) -> Path:
    """Save a volume as NIfTI with identity affine."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(volume.astype(dtype), np.eye(4)), str(path))
    return path


def save_tiff(volume: np.ndarray, path: str | Path) -> Path:
    """Save a volume as TIFF."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(path), volume)
    return path

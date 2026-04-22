"""Helpers for the destripe demo notebook."""

from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import tifffile

from .paths import find_repo_root, setup_environment
from .visualization import render_figure


SUPPORTED_INPUT_EXTENSIONS = (".tif", ".tiff", ".nii", ".nii.gz", ".mha", ".mhd", ".png", ".jpg", ".jpeg")


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


def _image_stem(path: str | Path) -> str:
    path = Path(path)
    if path.name.lower().endswith(".nii.gz"):
        return path.name[:-7]
    return path.stem


def _read_image_any(path: str | Path) -> np.ndarray:
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
    elif name.endswith((".png", ".jpg", ".jpeg")):
        try:
            from PIL import Image
        except ImportError as exc:
            raise ImportError("Pillow is required to read PNG/JPG files") from exc
        arr = np.asarray(Image.open(path))
    else:
        raise ValueError(f"Unsupported destripe input file: {path}")
    arr = np.asarray(arr)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{path.name} contains NaN or infinite values")
    return arr


def _rgb_to_gray(arr: np.ndarray) -> np.ndarray:
    rgb = arr[..., :3].astype(np.float32)
    return (0.2989 * rgb[..., 0]) + (0.5870 * rgb[..., 1]) + (0.1140 * rgb[..., 2])


def _find_channel_axis(arr: np.ndarray) -> int | None:
    if arr.ndim not in (3, 4):
        return None
    candidates = [axis for axis, size in enumerate(arr.shape) if size <= 4]
    if not candidates:
        return None
    for axis in (arr.ndim - 1, 0, 1):
        if axis in candidates:
            return axis
    return candidates[0]


def _sanitize_destripe_input(arr: np.ndarray, path: Path | None = None) -> tuple[np.ndarray, list[str]]:
    warnings: list[str] = []
    shape = arr.shape
    name = path.name if path is not None else "input"
    if arr.ndim == 2:
        return arr.astype(np.float32), warnings
    if arr.ndim == 3:
        if shape[-1] in (1, 3, 4) and min(shape[0], shape[1]) > 8:
            if shape[-1] == 1:
                warnings.append(f"{name}: squeezed single-channel last axis.")
                return arr[..., 0].astype(np.float32), warnings
            warnings.append(f"{name}: converted RGB input to grayscale.")
            return _rgb_to_gray(arr), warnings
        if shape[0] in (1, 3, 4) and min(shape[1], shape[2]) > 8:
            moved = np.moveaxis(arr, 0, -1)
            if moved.shape[-1] == 1:
                warnings.append(f"{name}: squeezed single-channel first axis.")
                return moved[..., 0].astype(np.float32), warnings
            warnings.append(f"{name}: converted channel-first RGB input to grayscale.")
            return _rgb_to_gray(moved), warnings
        return arr.astype(np.float32), warnings
    if arr.ndim == 4:
        channel_axis = _find_channel_axis(arr)
        if channel_axis is None:
            raise ValueError(
                f"{name} has shape {shape}. Expected a 2D image, 3D volume, or input with a small channel axis."
            )
        channels = np.moveaxis(arr, channel_axis, -1)
        n_channels = channels.shape[-1]
        if n_channels == 1:
            warnings.append(f"{name}: squeezed single-channel axis {channel_axis}.")
            return np.asarray(channels[..., 0], dtype=np.float32), warnings
        if n_channels in (3, 4):
            warnings.append(f"{name}: converted {n_channels}-channel input to grayscale.")
            return _rgb_to_gray(channels), warnings
        raise ValueError(f"{name} has {n_channels} channels. Provide a grayscale image or volume.")
    raise ValueError(f"{name} must be 2D or 3D after channel cleanup; got shape {shape}")


def _middle_slice(arr: np.ndarray) -> np.ndarray:
    return arr if arr.ndim == 2 else arr[arr.shape[0] // 2]


def _crop_center(arr: np.ndarray, frac: float = 0.35) -> np.ndarray:
    image = _middle_slice(arr)
    h, w = image.shape[:2]
    crop_h = max(32, int(h * frac))
    crop_w = max(32, int(w * frac))
    y0 = max(0, (h - crop_h) // 2)
    x0 = max(0, (w - crop_w) // 2)
    return image[y0 : y0 + crop_h, x0 : x0 + crop_w]


@dataclass
class DestripeDemoConfig:
    base_dir: Path | str | None = None
    storage_dir: Path | str | None = None
    data_dir: Path | str | None = None
    input_dir: Path | str = Path("destripe_demo/input")
    output_dir: Path | str = Path("destripe_demo/output")
    sigma: float = 8.0
    max_threshold: float = 12.0
    wavelet: str = "db3"
    level: int | None = None

    def environment_paths(self, base_dir: str | Path | None = None, verbose: bool = False) -> dict[str, Path]:
        return setup_environment(
            base_dir=base_dir or self.base_dir,
            storage_dir=self.storage_dir,
            data_dir=self.data_dir,
            verbose=verbose,
        )

    def resolve(self, base_dir: str | Path | None = None) -> "DestripeDemoConfig":
        paths = self.environment_paths(base_dir, verbose=False)
        resolved = DestripeDemoConfig(**self.__dict__)
        resolved.base_dir = paths["base"]
        resolved.storage_dir = paths["storage"]
        resolved.data_dir = paths["data"]
        resolved.input_dir = _resolve_under_root(resolved.input_dir, paths["data"], "data")
        resolved.output_dir = _resolve_under_root(resolved.output_dir, paths["data"], "data")
        return resolved


@dataclass
class DestripeDemoState:
    config: DestripeDemoConfig
    used_synthetic_demo: bool
    input_path: Path
    output_path: Path
    input_array: np.ndarray
    clean_reference: np.ndarray | None = None
    filtered_array: np.ndarray | None = None
    warnings: list[str] = field(default_factory=list)

    def show(self) -> None:
        print(f"Workspace root:   {self.config.storage_dir}")
        print(f"Managed input:    {self.input_path}")
        print(f"Output image:     {self.output_path}")
        print(f"Demo mode:        {self.used_synthetic_demo}")
        print(f"Input shape:      {self.input_array.shape}")
        if self.warnings:
            print("\nWarnings:")
            for message in self.warnings:
                print(f"  - {message}")


def ensure_destripe_filter(repo_root: str | Path | None = None):
    """Import the Allen destripe filter, cloning the source if needed."""
    try:
        from aind_smartspim_destripe.filtering import log_space_fft_filtering

        return log_space_fft_filtering
    except ImportError:
        pass

    repo_root = find_repo_root(repo_root)
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--quiet", "PyWavelets", "scikit-image"],
        cwd=str(repo_root),
    )

    external_dir = Path(repo_root) / ".external" / "aind-smartspim-destripe"
    external_dir.parent.mkdir(parents=True, exist_ok=True)
    if not external_dir.exists():
        subprocess.check_call(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--quiet",
                "https://github.com/AllenNeuralDynamics/aind-smartspim-destripe.git",
                str(external_dir),
            ],
            cwd=str(repo_root),
        )

    code_dir = external_dir / "code"
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))

    from aind_smartspim_destripe.filtering import log_space_fft_filtering

    return log_space_fft_filtering


def generate_synthetic_striped_image(seed: int = 42, size: int = 512) -> tuple[np.ndarray, np.ndarray]:
    from scipy.ndimage import gaussian_filter

    np.random.seed(seed)
    yy, xx = np.mgrid[:size, :size]
    tissue = gaussian_filter(np.random.rand(size, size), sigma=10) * 150 + 45
    for _ in range(24):
        cx, cy = np.random.randint(35, size - 35, 2)
        radius = np.random.randint(8, 26)
        mask = ((xx - cx) ** 2 + (yy - cy) ** 2) < radius**2
        tissue[mask] += np.random.uniform(45, 110)
    tissue = gaussian_filter(tissue, sigma=0.8)
    tissue = np.clip(tissue, 0, 255).astype(np.float32)

    row = np.linspace(0, 2 * np.pi, size)
    stripe_profile = 6 * np.sin(row * 18) + 2 * np.sin(row * 43 + 0.7)
    striped = np.clip(tissue + stripe_profile[:, None], 0, 255).astype(np.float32)
    return tissue, striped


def prepare_destripe_demo(
    config: DestripeDemoConfig,
    input_path: str | Path | None = None,
    base_dir: str | Path | None = None,
    reset_workspace: bool = True,
) -> DestripeDemoState:
    config = config.resolve(base_dir)
    input_dir = Path(config.input_dir)
    output_dir = Path(config.output_dir)
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    if reset_workspace:
        for path in (input_dir, output_dir):
            for child in path.iterdir():
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()

    if input_path is None:
        clean, striped = generate_synthetic_striped_image()
        input_file = input_dir / "synthetic_striped_input.tif"
        output_file = output_dir / "synthetic_destriped_output.tif"
        tifffile.imwrite(str(input_file), striped.astype(np.float32))
        tifffile.imwrite(str(input_dir / "synthetic_clean_reference.tif"), clean.astype(np.float32))
        return DestripeDemoState(
            config=config,
            used_synthetic_demo=True,
            input_path=input_file,
            output_path=output_file,
            input_array=striped,
            clean_reference=clean,
        )

    original_path = Path(input_path)
    if not original_path.exists():
        raise FileNotFoundError(f"Destripe input not found: {original_path}")
    arr, warnings = _sanitize_destripe_input(_read_image_any(original_path), original_path)
    staged_name = f"{_image_stem(original_path)}.tif"
    input_file = input_dir / staged_name
    output_file = output_dir / f"{_image_stem(original_path)}_destriped.tif"
    tifffile.imwrite(str(input_file), arr.astype(np.float32))
    return DestripeDemoState(
        config=config,
        used_synthetic_demo=False,
        input_path=input_file,
        output_path=output_file,
        input_array=arr,
        warnings=warnings,
    )


def _apply_single_slice(
    image: np.ndarray,
    filter_fn,
    sigma: float,
    max_threshold: float,
    wavelet: str,
    level: int | None,
) -> np.ndarray:
    filtered = filter_fn(
        image.astype(np.float32),
        wavelet=wavelet,
        level=level,
        sigma=sigma,
        max_threshold=max_threshold,
    )
    filtered = filtered + (np.median(image) - np.median(filtered))
    return np.clip(filtered, float(np.min(image)), float(np.max(image))).astype(np.float32)


def run_destripe(
    state: DestripeDemoState,
    filter_fn=None,
    base_dir: str | Path | None = None,
) -> Path:
    if filter_fn is None:
        filter_fn = ensure_destripe_filter(base_dir or state.config.base_dir)

    source = state.input_array
    if source.ndim == 2:
        filtered = _apply_single_slice(
            source,
            filter_fn,
            sigma=state.config.sigma,
            max_threshold=state.config.max_threshold,
            wavelet=state.config.wavelet,
            level=state.config.level,
        )
    else:
        filtered = np.empty_like(source, dtype=np.float32)
        for index in range(source.shape[0]):
            filtered[index] = _apply_single_slice(
                source[index],
                filter_fn,
                sigma=state.config.sigma,
                max_threshold=state.config.max_threshold,
                wavelet=state.config.wavelet,
                level=state.config.level,
            )

    tifffile.imwrite(str(state.output_path), filtered.astype(np.float32))
    state.filtered_array = filtered
    return state.output_path


def rmse(image: np.ndarray, reference: np.ndarray) -> float:
    return float(np.sqrt(np.mean((image.astype(np.float32) - reference.astype(np.float32)) ** 2)))


def row_bias_std(image: np.ndarray, reference: np.ndarray) -> float:
    input_slice = _middle_slice(image)
    reference_slice = _middle_slice(reference)
    return float(np.std(np.median(input_slice - reference_slice, axis=1)))


def show_destripe_overview(state: DestripeDemoState, dpi: int = 200) -> None:
    if state.filtered_array is None:
        raise ValueError("Run run_destripe(...) before showing results.")
    input_slice = _middle_slice(state.input_array)
    filtered_slice = _middle_slice(state.filtered_array)
    panels: list[tuple[str, np.ndarray]] = []
    if state.clean_reference is not None:
        panels.append(("Synthetic clean reference", _middle_slice(state.clean_reference)))
    panels.extend(
        [
            ("Input with stripes", input_slice),
            ("Destriped output", filtered_slice),
        ]
    )
    vmin, vmax = np.percentile(input_slice, [1, 99.5])
    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 6), dpi=dpi)
    axes = np.atleast_1d(axes)
    for axis, (title, image) in zip(axes, panels):
        axis.imshow(image, cmap="gray", vmin=vmin, vmax=vmax)
        axis.set_title(title)
        axis.axis("off")
    plt.tight_layout()
    render_figure(fig)


def show_destripe_crop(state: DestripeDemoState, dpi: int = 200) -> None:
    if state.filtered_array is None:
        raise ValueError("Run run_destripe(...) before showing results.")
    panels: list[tuple[str, np.ndarray]] = []
    if state.clean_reference is not None:
        panels.append(("Reference crop", _crop_center(state.clean_reference)))
    panels.extend(
        [
            ("Input crop", _crop_center(state.input_array)),
            ("Destriped crop", _crop_center(state.filtered_array)),
        ]
    )
    input_crop = panels[-2][1]
    vmin, vmax = np.percentile(input_crop, [1, 99.5])
    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 6), dpi=dpi)
    axes = np.atleast_1d(axes)
    for axis, (title, image) in zip(axes, panels):
        axis.imshow(image, cmap="gray", vmin=vmin, vmax=vmax)
        axis.set_title(title)
        axis.axis("off")
    plt.tight_layout()
    render_figure(fig)

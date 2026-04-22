"""Path and environment setup for repo-local or external storage."""

from __future__ import annotations

import os
from pathlib import Path


ENV_STORAGE_DIR = "DEEPBRANCHAI_STORAGE_DIR"
ENV_DATA_DIR = "DEEPBRANCHAI_DATA_DIR"
ENV_WEIGHTS_DIR = "DEEPBRANCHAI_WEIGHTS_DIR"
ENV_TMP_DIR = "DEEPBRANCHAI_TMP_DIR"
ENV_NNUNET_RAW = "DEEPBRANCHAI_NNUNET_RAW"
ENV_NNUNET_PREPROCESSED = "DEEPBRANCHAI_NNUNET_PREPROCESSED"
ENV_NNUNET_RESULTS = "DEEPBRANCHAI_NNUNET_RESULTS"


def find_repo_root(start: str | Path | None = None) -> Path:
    """Find the repository root by walking upward from ``start``."""
    current = Path.cwd().resolve() if start is None else Path(start).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "README.md").exists() and (
            (candidate / "deepbranchai_utils.py").exists()
            or (candidate / "deepbranchai").is_dir()
        ):
            return candidate
    raise RuntimeError("Could not find repository root")


def _env_path(name: str) -> str | None:
    value = os.environ.get(name)
    return value if value else None


def _resolve_path(value: str | Path | None, default: Path, base: Path) -> Path:
    path = Path(default if value is None else value)
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def setup_environment(
    base_dir: str | Path | None = None,
    storage_dir: str | Path | None = None,
    data_dir: str | Path | None = None,
    weights_dir: str | Path | None = None,
    tmp_dir: str | Path | None = None,
    nnunet_raw: str | Path | None = None,
    nnunet_preprocessed: str | Path | None = None,
    nnunet_results: str | Path | None = None,
    verbose: bool = True,
) -> dict[str, Path]:
    """Create storage folders and set nnU-Net environment variables.

    By default, everything lives under the repository. For large datasets, pass
    ``storage_dir`` or set ``DEEPBRANCHAI_STORAGE_DIR`` to keep data, weights,
    tmp files, and nnU-Net folders on an external/data drive.
    """
    base = find_repo_root(base_dir)
    storage = _resolve_path(
        storage_dir or _env_path(ENV_STORAGE_DIR),
        base,
        base,
    )
    paths = {
        "base": base,
        "storage": storage,
        "weights": _resolve_path(weights_dir or _env_path(ENV_WEIGHTS_DIR), storage / "weights", base),
        "data": _resolve_path(data_dir or _env_path(ENV_DATA_DIR), storage / "data", base),
        "tmp": _resolve_path(tmp_dir or _env_path(ENV_TMP_DIR), storage / "tmp", base),
        "nnUNet_raw": _resolve_path(nnunet_raw or _env_path(ENV_NNUNET_RAW), storage / "nnUNet_raw", base),
        "nnUNet_preprocessed": _resolve_path(
            nnunet_preprocessed or _env_path(ENV_NNUNET_PREPROCESSED),
            storage / "nnUNet_preprocessed",
            base,
        ),
        "nnUNet_results": _resolve_path(
            nnunet_results or _env_path(ENV_NNUNET_RESULTS),
            storage / "nnUNet_results",
            base,
        ),
    }

    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    os.environ["nnUNet_raw"] = str(paths["nnUNet_raw"])
    os.environ["nnUNet_preprocessed"] = str(paths["nnUNet_preprocessed"])
    os.environ["nnUNet_results"] = str(paths["nnUNet_results"])
    # Keep older notebooks and helper scripts aligned with the same storage root.
    os.environ["nnUNet_raw_data_base"] = str(paths["nnUNet_raw"])
    os.environ["RESULTS_FOLDER"] = str(paths["nnUNet_results"])
    os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"

    if verbose:
        print(f"Base directory:       {paths['base']}")
        print(f"Storage directory:    {paths['storage']}")
        print(f"Data directory:       {paths['data']}")
        print(f"Weights directory:    {paths['weights']}")
        print(f"Tmp directory:        {paths['tmp']}")
        print(f"nnUNet_raw:           {paths['nnUNet_raw']}")
        print(f"nnUNet_preprocessed:  {paths['nnUNet_preprocessed']}")
        print(f"nnUNet_results:       {paths['nnUNet_results']}")

    return paths


def check_gpu() -> bool:
    """Verify CUDA availability and print GPU details."""
    import torch

    print(f"PyTorch version:  {torch.__version__}")
    print(f"CUDA available:   {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version:     {torch.version.cuda}")
        print(f"GPU:              {torch.cuda.get_device_name(0)}")
        print(f"VRAM:             {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print()
        print("*** WARNING: No CUDA GPU detected. Training/inference will be extremely slow. ***")
        print("Install CUDA-enabled PyTorch:")
        print("  pip install torch --index-url https://download.pytorch.org/whl/cu118")
    return torch.cuda.is_available()

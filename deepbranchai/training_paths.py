"""Training dataset path helpers with neutral directory names."""

from __future__ import annotations

from pathlib import Path


def get_two_d_training_dir(data_dir: str | Path) -> Path:
    """Return the default 2D training input directory."""
    return Path(data_dir) / "training_2d" / "TEST_TRAIN"


def get_preprocessing_fix_dir(data_dir: str | Path) -> Path:
    """Return the default preprocessing fix directory."""
    return Path(data_dir) / "training_redo" / "fix"


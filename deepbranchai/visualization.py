"""Visualization helpers for notebooks."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .image_io import load_volume


def middle_slice(volume: np.ndarray) -> np.ndarray:
    return volume[volume.shape[0] // 2]


def render_figure(fig) -> None:
    try:
        from IPython.display import display

        display(fig)
    except Exception:
        plt.show()
    finally:
        plt.close(fig)


def show_raw_mask_pair(
    raw_path: str | Path,
    mask_path: str | Path,
    channel_index: int | None = None,
    dpi: int = 100,
) -> None:
    raw = load_volume(raw_path, channel_index=channel_index)
    mask = load_volume(mask_path, channel_index=channel_index)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=dpi)
    axes[0].imshow(middle_slice(raw), cmap="gray")
    axes[0].set_title("Raw middle slice")
    axes[0].axis("off")

    axes[1].imshow(middle_slice(mask), cmap="gray")
    axes[1].set_title("Ground-truth mask")
    axes[1].axis("off")

    plt.tight_layout()
    render_figure(fig)


def show_validation_prediction_slice(
    raw_path: str | Path,
    ground_truth_path: str | Path,
    prediction_path: str | Path,
    channel_index: int | None = None,
    dpi: int = 200,
) -> None:
    raw = load_volume(raw_path, channel_index=channel_index)
    ground_truth = load_volume(ground_truth_path, channel_index=channel_index)
    prediction = load_volume(prediction_path, channel_index=channel_index)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=dpi)
    for axis, image, title in (
        (axes[0], middle_slice(raw), "Validation raw"),
        (axes[1], middle_slice(ground_truth), "Validation ground truth"),
        (axes[2], middle_slice(prediction), "Validation prediction"),
    ):
        axis.imshow(image, cmap="gray")
        axis.set_title(title)
        axis.axis("off")

    plt.tight_layout()
    render_figure(fig)


def show_raw_prediction_slice(
    raw_path: str | Path,
    prediction_path: str | Path,
    channel_index: int | None = None,
    dpi: int = 200,
) -> None:
    raw = load_volume(raw_path, channel_index=channel_index)
    prediction = load_volume(prediction_path, channel_index=channel_index)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=dpi)
    for axis, image, title in (
        (axes[0], middle_slice(raw), "Inference raw"),
        (axes[1], middle_slice(prediction), "Inference prediction"),
    ):
        axis.imshow(image, cmap="gray")
        axis.set_title(title)
        axis.axis("off")

    plt.tight_layout()
    render_figure(fig)

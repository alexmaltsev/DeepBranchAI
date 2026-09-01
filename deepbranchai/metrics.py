"""Volume-wise binary segmentation metrics used by DeepBranchAI."""

from __future__ import annotations

import math
from collections.abc import Mapping

import cc3d
import numpy as np
from skimage.morphology import skeletonize


def _as_binary(volume: np.ndarray) -> np.ndarray:
    array = np.asarray(volume)
    if array.ndim != 3:
        raise ValueError(f"Expected a 3D volume, received shape {array.shape}")
    return array > 0


def _divide(numerator: float, denominator: float, *, empty_value: float = math.nan) -> float:
    return float(numerator / denominator) if denominator else empty_value


def _component_count(mask: np.ndarray) -> int:
    if not mask.any():
        return 0
    _, count = cc3d.connected_components(mask.astype(np.uint8), connectivity=26, return_N=True)
    return int(count)


def cldice_score(prediction: np.ndarray, reference: np.ndarray) -> float:
    """Compute centerline Dice from skeletonized 3D masks."""
    prediction = _as_binary(prediction)
    reference = _as_binary(reference)
    prediction_skeleton = skeletonize(prediction).astype(bool, copy=False)
    reference_skeleton = skeletonize(reference).astype(bool, copy=False)

    prediction_count = int(prediction_skeleton.sum())
    reference_count = int(reference_skeleton.sum())
    if prediction_count == 0 and reference_count == 0:
        return 1.0
    if prediction_count == 0 or reference_count == 0:
        return 0.0

    topology_precision = float(np.logical_and(prediction_skeleton, reference).sum() / prediction_count)
    topology_sensitivity = float(np.logical_and(reference_skeleton, prediction).sum() / reference_count)
    denominator = topology_precision + topology_sensitivity
    return float(2 * topology_precision * topology_sensitivity / denominator) if denominator else 0.0


def compute_binary_volume_metrics(
    prediction: np.ndarray,
    reference: np.ndarray,
) -> Mapping[str, float | int]:
    """Compute the accepted paper's metrics on one assembled 3D prediction."""
    prediction = _as_binary(prediction)
    reference = _as_binary(reference)
    if prediction.shape != reference.shape:
        raise ValueError(f"Prediction shape {prediction.shape} does not match reference shape {reference.shape}")

    tp = int(np.logical_and(prediction, reference).sum())
    tn = int(np.logical_and(~prediction, ~reference).sum())
    fp = int(np.logical_and(prediction, ~reference).sum())
    fn = int(np.logical_and(~prediction, reference).sum())
    total = tp + tn + fp + fn
    prediction_positive = tp + fp
    reference_positive = tp + fn

    precision = _divide(tp, prediction_positive, empty_value=1.0 if reference_positive == 0 else 0.0)
    sensitivity = _divide(tp, reference_positive, empty_value=1.0 if prediction_positive == 0 else 0.0)
    specificity = _divide(tn, tn + fp)
    accuracy = _divide(tp + tn, total)
    dice = _divide(2 * tp, 2 * tp + fp + fn, empty_value=1.0)
    avd_percent = (
        abs(prediction_positive - reference_positive) / reference_positive * 100
        if reference_positive
        else (0.0 if prediction_positive == 0 else math.nan)
    )

    observed_agreement = accuracy
    expected_agreement = _divide(
        prediction_positive * reference_positive + (tn + fn) * (tn + fp),
        total * total,
    )
    kappa = _divide(observed_agreement - expected_agreement, 1 - expected_agreement)

    prediction_components = _component_count(prediction)
    reference_components = _component_count(reference)
    return {
        "precision": precision,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "accuracy": accuracy,
        "dice": dice,
        "cldice": cldice_score(prediction, reference),
        "prediction_components": prediction_components,
        "reference_components": reference_components,
        "abs_cc_error": abs(prediction_components - reference_components),
        "avd_percent": float(avd_percent),
        "kappa": kappa,
    }

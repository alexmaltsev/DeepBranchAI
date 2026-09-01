from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import tifffile
import torch

STYLE_ROOT = Path(__file__).resolve().parent.parent / "soltaninejad_style"
sys.path.append(str(STYLE_ROOT))

from data import NpyVolume, TiffVolume, block_max_binary, cached_paths
from metrics import full_volume_metrics
from paths import DEFAULT_SOURCE_ROOT, fold_case_records, read_json, write_json
from unet_data import parse_patch_size_zyx
from unet_model import UNet3D


DEFAULT_EXPERIMENT_ROOT = Path(__file__).resolve().parents[4] / "transfer" / "unet3d_baseline"


def load_model(checkpoint: Path, device: torch.device) -> tuple[UNet3D, dict]:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model_args = payload.get("model_args", {})
    model = UNet3D(**model_args).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, payload


def axis_starts(length: int, patch_size: int, stride: int) -> list[int]:
    if length <= patch_size:
        return [0]
    starts = list(range(0, max(length - patch_size + 1, 1), stride))
    last = length - patch_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def grid_starts(
    shape: tuple[int, int, int],
    patch_size_zyx: tuple[int, int, int],
    stride_zyx: tuple[int, int, int],
) -> list[tuple[int, int, int]]:
    return [
        (z, y, x)
        for z in axis_starts(int(shape[0]), patch_size_zyx[0], stride_zyx[0])
        for y in axis_starts(int(shape[1]), patch_size_zyx[1], stride_zyx[1])
        for x in axis_starts(int(shape[2]), patch_size_zyx[2], stride_zyx[2])
    ]


def downsample_label_volume(label_path: Path, factor: int) -> np.ndarray:
    volume = TiffVolume(label_path)
    shape = np.asarray(volume.shape, dtype=np.int64)
    if factor == 1:
        out = np.zeros(tuple(shape.tolist()), dtype=np.uint8)
        try:
            for z in range(shape[0]):
                out[z] = volume.read_patch((z, 0, 0), (1, int(shape[1]), int(shape[2])))[0] > 0
        finally:
            volume.close()
        return out

    cropped = shape // factor * factor
    out_shape = cropped // factor
    out = np.zeros(tuple(out_shape.tolist()), dtype=np.uint8)
    try:
        for oz in range(out_shape[0]):
            start_z = int(oz * factor)
            slab = volume.read_patch((start_z, 0, 0), (factor, int(cropped[1]), int(cropped[2])), fill_value=0)
            out[oz] = block_max_binary(slab, factor)[0]
    finally:
        volume.close()
    return out


def predict_case_cached(
    model: UNet3D,
    raw_cache_path: Path,
    device: torch.device,
    patch_size_zyx: tuple[int, int, int],
    stride_zyx: tuple[int, int, int],
    intensity_scale: float,
    threshold: float,
    batch_size: int,
    amp: bool,
    save_probabilities: bool,
    progress_label: str,
) -> tuple[np.ndarray | None, np.ndarray]:
    raw_volume = NpyVolume(raw_cache_path)
    out_shape = raw_volume.shape
    prob_sum = np.zeros(out_shape, dtype=np.float32)
    counts = np.zeros(out_shape, dtype=np.uint16)
    starts = grid_starts(out_shape, patch_size_zyx, stride_zyx)
    total = len(starts)
    last_report = time.time()

    for start_index in range(0, total, batch_size):
        batch_starts = starts[start_index : start_index + batch_size]
        patches = []
        for start in batch_starts:
            raw_patch = raw_volume.read_patch(start, patch_size_zyx, fill_value=0)
            raw_patch = np.clip(raw_patch.astype(np.float32, copy=False) / intensity_scale, 0.0, 1.0)
            patches.append(raw_patch[None, ...])

        tensor = torch.from_numpy(np.stack(patches, axis=0).copy()).to(device)
        with torch.inference_mode():
            with torch.amp.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                batch_probs = torch.sigmoid(model(tensor).logits).detach().cpu().numpy()[:, 0]

        for item_index, (z0, y0, x0) in enumerate(batch_starts):
            z1 = min(z0 + patch_size_zyx[0], int(out_shape[0]))
            y1 = min(y0 + patch_size_zyx[1], int(out_shape[1]))
            x1 = min(x0 + patch_size_zyx[2], int(out_shape[2]))
            cropped = batch_probs[item_index, : z1 - z0, : y1 - y0, : x1 - x0]
            prob_sum[z0:z1, y0:y1, x0:x1] += cropped
            counts[z0:z1, y0:y1, x0:x1] += 1

        now = time.time()
        done = min(start_index + batch_size, total)
        if now - last_report >= 30.0 or done == total:
            print(f"{progress_label}: {done}/{total} patches")
            last_report = now

    probabilities = prob_sum / np.maximum(counts.astype(np.float32), 1.0)
    mask = (probabilities >= threshold).astype(np.uint8, copy=False)
    if save_probabilities:
        return probabilities.astype(np.float16, copy=False), mask
    return None, mask


def write_rows_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def infer_fold(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    checkpoint = args.checkpoint
    if checkpoint is None:
        checkpoint = args.experiment_root / f"fold_{args.fold}" / "checkpoints" / "checkpoint_best.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")

    model, payload = load_model(checkpoint, device)
    _, val_records, _ = fold_case_records(args.source_root, args.experiment_root, args.fold)
    if args.case_id:
        wanted = set(args.case_id)
        val_records = [record for record in val_records if record.nnunet_id in wanted or record.source_id in wanted]
        if not val_records:
            raise ValueError(f"No validation records matched --case-id values: {sorted(wanted)}")

    out_root = args.experiment_root / f"fold_{args.fold}" / "validation_predictions"
    out_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for record in val_records:
        metric_path = out_root / f"{record.nnunet_id}_{record.source_id}_metrics.json"
        mask_path = out_root / f"{record.nnunet_id}_{record.source_id}_mask.tif"
        prob_path = out_root / f"{record.nnunet_id}_{record.source_id}_prob.tif"
        if args.skip_existing and metric_path.exists() and mask_path.exists():
            print(f"Skipping existing {record.nnunet_id} {record.source_id}")
            rows.append(read_json(metric_path))
            continue

        raw_cache, label_cache = cached_paths(args.cache_dir, record)
        if not raw_cache.exists():
            raise FileNotFoundError(f"Missing downsampled raw cache for {record.nnunet_id}: {raw_cache}")
        print(f"Predicting {record.nnunet_id} {record.source_id}")
        prob, pred = predict_case_cached(
            model,
            raw_cache,
            device,
            patch_size_zyx=args.patch_size_zyx,
            stride_zyx=args.stride_zyx,
            intensity_scale=args.intensity_scale,
            threshold=args.threshold,
            batch_size=args.batch_size,
            amp=args.amp_infer,
            save_probabilities=args.save_prob,
            progress_label=f"fold={args.fold} {record.nnunet_id}",
        )
        target = np.load(label_cache, mmap_mode="r") if label_cache.exists() else downsample_label_volume(record.label, args.downsample_factor)
        metrics = full_volume_metrics(pred, target)
        if args.save_prob and prob is not None:
            tifffile.imwrite(prob_path, prob, compression="zlib")
        tifffile.imwrite(mask_path, pred, compression="zlib")
        row = {"case_id": record.nnunet_id, "source_id": record.source_id, **metrics}
        rows.append(row)
        write_json(metric_path, row)

    mean_metrics = {}
    numeric_keys = [key for key in rows[0].keys() if key not in {"case_id", "source_id"}] if rows else []
    for key in numeric_keys:
        values = [float(row[key]) for row in rows if np.isfinite(float(row[key]))]
        mean_metrics[f"mean_{key}"] = float(np.mean(values)) if values else float("nan")
    summary = {
        "fold": args.fold,
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": payload.get("epoch"),
        "cases": rows,
        **mean_metrics,
    }
    write_json(args.experiment_root / f"fold_{args.fold}" / "validation_metrics.json", summary)
    write_rows_csv(args.experiment_root / f"fold_{args.fold}" / "validation_metrics.csv", rows)
    print(f"Wrote {args.experiment_root / f'fold_{args.fold}' / 'validation_metrics.json'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run full-volume validation for the Plant CT 3D U-Net baseline.")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu", "mps"])
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--patch-size-zyx", type=int, nargs=3, default=None, metavar=("Z", "Y", "X"))
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--stride-zyx", type=int, nargs=3, default=None, metavar=("Z", "Y", "X"))
    parser.add_argument("--downsample-factor", type=int, default=3)
    parser.add_argument("--intensity-scale", type=float, default=255.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_EXPERIMENT_ROOT / "downsample3x_cache")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--amp-infer", action="store_true")
    parser.add_argument("--save-prob", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--case-id", action="append", default=[])
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.source_root = args.source_root.resolve()
    args.experiment_root = args.experiment_root.resolve()
    args.cache_dir = args.cache_dir.resolve()
    args.patch_size_zyx = tuple(args.patch_size_zyx) if args.patch_size_zyx else (args.patch_size, args.patch_size, args.patch_size)
    args.stride_zyx = tuple(args.stride_zyx) if args.stride_zyx else (args.stride, args.stride, args.stride)
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if any(stride < 1 for stride in args.stride_zyx):
        raise ValueError("--stride/--stride-zyx values must be at least 1")
    if any(stride > patch for stride, patch in zip(args.stride_zyx, args.patch_size_zyx)):
        raise ValueError("--stride values must be less than or equal to patch dimensions so every voxel is covered")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    infer_fold(args)


if __name__ == "__main__":
    main()

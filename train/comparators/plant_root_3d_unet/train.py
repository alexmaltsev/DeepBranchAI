from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

STYLE_ROOT = Path(__file__).resolve().parent.parent / "soltaninejad_style"
sys.path.append(str(STYLE_ROOT))

from data import ensure_center_indexes, ensure_downsample_cache
from metrics import confusion_from_logits, dice_loss_from_logits
from paths import DEFAULT_SOURCE_ROOT, fold_case_records, now, write_json
from unet_data import PlantRootUNetPatchDataset, collate_unet_patch_samples
from unet_model import UNet3D, count_parameters


DEFAULT_EXPERIMENT_ROOT = Path(__file__).resolve().parents[4] / "transfer" / "unet3d_baseline"


def set_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def append_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def jsonable_args(args: argparse.Namespace) -> dict:
    out = {}
    for key, value in vars(args).items():
        out[key] = str(value) if isinstance(value, Path) else value
    return out


def make_loader(args: argparse.Namespace, records, split: str) -> DataLoader:
    samples = args.train_samples_per_epoch if split == "train" else args.val_samples
    dataset = PlantRootUNetPatchDataset(
        records=records,
        index_dir=args.experiment_root / "center_index",
        samples_per_epoch=samples,
        patch_size_zyx=args.patch_size_zyx,
        downsample_factor=args.downsample_factor,
        intensity_scale=args.intensity_scale,
        random_background_probability=args.random_background_probability,
        random_offset=args.random_offset,
        seed=args.seed + (0 if split == "train" else 100000),
        cache_dir=None if args.no_cache else args.cache_dir,
    )
    batch_size = args.train_batch if split == "train" else args.valid_batch
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device == "cuda",
        collate_fn=collate_unet_patch_samples,
    )


def segmentation_loss(logits: torch.Tensor, target: torch.Tensor, loss_kind: str) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target)
    if loss_kind == "bce":
        return bce
    if loss_kind == "bce_dice":
        return bce + dice_loss_from_logits(logits, target)
    raise ValueError(f"Unsupported loss kind: {loss_kind}")


def save_checkpoint(
    path: Path,
    model: UNet3D,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_dice: float,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "best_dice": best_dice,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "model_args": {
                "in_channels": 1,
                "out_channels": 1,
                "base_channels": args.base_channels,
                "levels": args.levels,
                "norm": args.norm,
            },
            "train_args": jsonable_args(args),
        },
        path,
    )


def load_checkpoint(path: Path, model: UNet3D, optimizer: torch.optim.Optimizer, device: torch.device) -> tuple[int, float]:
    payload = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state"])
    optimizer.load_state_dict(payload["optimizer_state"])
    return int(payload["epoch"]) + 1, float(payload.get("best_dice", 0.0))


def step_epoch(
    model: UNet3D,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, float | int]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    count = 0
    finite_loss_count = 0
    nonfinite_batches = 0
    tp = fp = tn = fn = 0

    for batch in loader:
        inputs = batch["input"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
            logits = model(inputs).logits
            loss = segmentation_loss(logits, target, args.loss)

        finite_loss = bool(torch.isfinite(loss).detach().cpu())
        if training and finite_loss:
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        elif not finite_loss:
            nonfinite_batches += 1

        batch_size = int(inputs.shape[0])
        if finite_loss:
            total_loss += float(loss.detach().cpu()) * batch_size
            finite_loss_count += batch_size
        count += batch_size

        conf = confusion_from_logits(logits, target, threshold=args.threshold)
        tp += conf.tp
        fp += conf.fp
        tn += conf.tn
        fn += conf.fn

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    dice = (2 * tp) / max(2 * tp + fp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    return {
        "loss": total_loss / max(finite_loss_count, 1) if finite_loss_count else float("nan"),
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "samples": count,
        "nonfinite_batches": nonfinite_batches,
    }


def train_fold(args: argparse.Namespace) -> None:
    if any(size % (2 ** (args.levels - 1)) != 0 for size in args.patch_size_zyx):
        raise ValueError(
            f"patch-size-zyx={args.patch_size_zyx} must be divisible by {2 ** (args.levels - 1)} "
            f"for levels={args.levels}"
        )
    set_deterministic(args.seed + args.fold)
    train_records, val_records, split_manifest = fold_case_records(args.source_root, args.experiment_root, args.fold)
    fold_root = args.experiment_root / f"fold_{args.fold}"
    fold_root.mkdir(parents=True, exist_ok=True)
    write_json(fold_root / "split_manifest.json", split_manifest)

    index_records = train_records + val_records
    if args.prepare_cache_only or not args.no_build_cache:
        ensure_downsample_cache(index_records, args.cache_dir, factor=args.downsample_factor, force=args.force_cache)
    if args.prepare_cache_only:
        print(f"Prepared downsampled cache for fold {args.fold} under {args.cache_dir}")
        return
    if args.prepare_index_only or not args.no_build_index:
        ensure_center_indexes(
            index_records,
            args.experiment_root / "center_index",
            max_points_per_case=args.max_points_per_case,
            force=args.force_index,
        )
    if args.prepare_index_only:
        print(f"Prepared center indexes for fold {args.fold} under {args.experiment_root / 'center_index'}")
        return

    device = torch.device(args.device)
    model = UNet3D(base_channels=args.base_channels, levels=args.levels, norm=args.norm).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    start_epoch = 1
    best_dice = 0.0
    latest = fold_root / "checkpoints" / "checkpoint_latest.pt"
    if args.resume and latest.exists():
        start_epoch, best_dice = load_checkpoint(latest, model, optimizer, device)
        print(f"Resumed fold {args.fold} from epoch {start_epoch}")

    run_manifest = {
        "created": now(),
        "fold": args.fold,
        "train_cases": [record.nnunet_id for record in train_records],
        "validation_cases": [record.nnunet_id for record in val_records],
        "parameter_count": count_parameters(model),
        "args": jsonable_args(args),
    }
    write_json(fold_root / "run_manifest.json", run_manifest)

    train_loader = make_loader(args, train_records, "train")
    val_loader = make_loader(args, val_records, "valid")
    log_path = fold_root / "logs" / "training_log.csv"

    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = step_epoch(model, train_loader, optimizer, scaler, device, args)
        with torch.no_grad():
            val_metrics = step_epoch(model, val_loader, None, scaler, device, args)

        row = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        append_csv(log_path, row)
        save_checkpoint(latest, model, optimizer, epoch, best_dice, args)
        if val_metrics["dice"] >= best_dice:
            best_dice = float(val_metrics["dice"])
            save_checkpoint(fold_root / "checkpoints" / "checkpoint_best.pt", model, optimizer, epoch, best_dice, args)
        if args.save_every and epoch % args.save_every == 0:
            save_checkpoint(fold_root / "checkpoints" / f"checkpoint_epoch_{epoch:03d}.pt", model, optimizer, epoch, best_dice, args)

        print(
            f"fold={args.fold} epoch={epoch:03d} "
            f"train_loss={train_metrics['loss']:.6f} train_dice={train_metrics['dice']:.6f} "
            f"val_loss={val_metrics['loss']:.6f} val_dice={val_metrics['dice']:.6f} "
            f"best={best_dice:.6f}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a conventional 3D U-Net baseline on Plant CT folds.")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu", "mps"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--train-batch", type=int, default=1)
    parser.add_argument("--valid-batch", type=int, default=1)
    parser.add_argument("--train-samples-per-epoch", type=int, default=1000)
    parser.add_argument("--val-samples", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--patch-size-zyx", type=int, nargs=3, default=None, metavar=("Z", "Y", "X"))
    parser.add_argument("--downsample-factor", type=int, default=3)
    parser.add_argument("--intensity-scale", type=float, default=255.0)
    parser.add_argument("--random-background-probability", type=float, default=0.30)
    parser.add_argument("--random-offset", type=int, default=8)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--levels", type=int, default=4)
    parser.add_argument("--norm", choices=["batch", "group", "instance"], default="instance")
    parser.add_argument("--loss", choices=["bce", "bce_dice"], default="bce_dice")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--grad-clip", type=float, default=12.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_EXPERIMENT_ROOT / "downsample3x_cache")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--no-build-cache", action="store_true")
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--prepare-cache-only", action="store_true")
    parser.add_argument("--no-build-index", action="store_true")
    parser.add_argument("--force-index", action="store_true")
    parser.add_argument("--prepare-index-only", action="store_true")
    parser.add_argument("--max-points-per-case", type=int, default=50000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.source_root = args.source_root.resolve()
    args.experiment_root = args.experiment_root.resolve()
    args.cache_dir = args.cache_dir.resolve()
    args.patch_size_zyx = tuple(args.patch_size_zyx) if args.patch_size_zyx else (args.patch_size, args.patch_size, args.patch_size)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    train_fold(args)


if __name__ == "__main__":
    main()

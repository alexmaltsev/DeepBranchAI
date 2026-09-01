from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

STYLE_ROOT = Path(__file__).resolve().parent.parent / "soltaninejad_style"
sys.path.append(str(STYLE_ROOT))

from paths import DEFAULT_SOURCE_ROOT, fold_case_records, load_or_create_split_manifest, records_by_id, write_json
from unet_data import PlantRootUNetPatchDataset
from unet_model import UNet3D, count_parameters


DEFAULT_EXPERIMENT_ROOT = Path(__file__).resolve().parents[4] / "transfer" / "unet3d_baseline"


def validate(args: argparse.Namespace) -> dict:
    source_root = args.source_root.resolve()
    experiment_root = args.experiment_root.resolve()
    cache_dir = args.cache_dir.resolve()
    records = records_by_id(source_root)
    manifest = load_or_create_split_manifest(source_root, experiment_root)
    fold_summaries = []
    for fold in range(5):
        train_records, val_records, _ = fold_case_records(source_root, experiment_root, fold)
        fold_summaries.append(
            {
                "fold": fold,
                "train_count": len(train_records),
                "validation_count": len(val_records),
                "train_ids": [record.nnunet_id for record in train_records],
                "validation_ids": [record.nnunet_id for record in val_records],
            }
        )

    result = {
        "source_root": str(source_root),
        "experiment_root": str(experiment_root),
        "cache_dir": str(cache_dir),
        "case_count": len(records),
        "split_source": manifest["split_source"],
        "folds": fold_summaries,
        "torch": {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        },
    }

    if not args.skip_model_smoke:
        device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
        model = UNet3D(base_channels=args.smoke_base_channels, levels=args.smoke_levels, norm=args.norm).to(device)
        model.eval()
        with torch.inference_mode():
            x = torch.zeros((1, 1, args.smoke_patch_size, args.smoke_patch_size, args.smoke_patch_size), dtype=torch.float32, device=device)
            logits = model(x).logits
        result["model_smoke"] = {
            "device": str(device),
            "parameters": count_parameters(model),
            "logits_shape": list(logits.shape),
        }

    if args.read_one_patch:
        first_fold_train, _, _ = fold_case_records(source_root, experiment_root, 0)
        use_cache = args.use_cache and all(
            (cache_dir / "raw" / f"{record.nnunet_id}.npy").exists() and (cache_dir / "labels" / f"{record.nnunet_id}.npy").exists()
            for record in first_fold_train[:1]
        )
        dataset = PlantRootUNetPatchDataset(
            records=first_fold_train[:1],
            index_dir=experiment_root / "center_index",
            samples_per_epoch=1,
            patch_size_zyx=args.patch_size_zyx,
            downsample_factor=args.downsample_factor,
            random_background_probability=1.0,
            seed=args.seed,
            cache_dir=cache_dir if use_cache else None,
        )
        sample = dataset[0]
        result["patch_smoke"] = {
            "used_cache": use_cache,
            "case_id": sample["case_id"],
            "source_id": sample["source_id"],
            "input_shape": list(sample["input"].shape),
            "target_shape": list(sample["target"].shape),
            "input_min": float(sample["input"].min()),
            "input_max": float(sample["input"].max()),
            "target_foreground": int(sample["target"].sum()),
        }

    write_json(experiment_root / "setup_validation.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate the Plant CT 3D U-Net baseline setup without training.")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_EXPERIMENT_ROOT / "downsample3x_cache")
    parser.add_argument("--device", default="cpu", choices=["cuda", "cpu", "mps"])
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--patch-size-zyx", type=int, nargs=3, default=None, metavar=("Z", "Y", "X"))
    parser.add_argument("--downsample-factor", type=int, default=3)
    parser.add_argument("--smoke-patch-size", type=int, default=32)
    parser.add_argument("--smoke-base-channels", type=int, default=4)
    parser.add_argument("--smoke-levels", type=int, default=3)
    parser.add_argument("--norm", choices=["batch", "group", "instance"], default="instance")
    parser.add_argument("--skip-model-smoke", action="store_true")
    parser.add_argument("--read-one-patch", action="store_true")
    parser.add_argument("--use-cache", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.patch_size_zyx = tuple(args.patch_size_zyx) if args.patch_size_zyx else (args.patch_size, args.patch_size, args.patch_size)
    result = validate(args)
    print(f"cases: {result['case_count']}")
    print(f"split source: {result['split_source']}")
    for fold in result["folds"]:
        print(f"fold {fold['fold']}: train={fold['train_count']} val={fold['validation_count']}")
    if "model_smoke" in result:
        smoke = result["model_smoke"]
        print(f"model smoke: logits={smoke['logits_shape']} params={smoke['parameters']} device={smoke['device']}")
    if "patch_smoke" in result:
        patch = result["patch_smoke"]
        print(
            "patch smoke: "
            f"case={patch['case_id']} input={patch['input_shape']} target={patch['target_shape']} "
            f"used_cache={patch['used_cache']}"
        )


if __name__ == "__main__":
    main()

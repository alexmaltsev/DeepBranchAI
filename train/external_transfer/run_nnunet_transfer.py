#!/usr/bin/env python3
"""Run matched scratch or DeepBranchAI-pretrained nnU-Net folds."""

from __future__ import annotations

import argparse
from pathlib import Path

from deepbranchai.custom_finetune import FinetuneConfig, ensure_pretrained_weights
from deepbranchai.nnunet_runner import train_nnunet_fold
from deepbranchai.paths import setup_environment


def parse_folds(value: str) -> list[int]:
    folds = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not folds or any(fold < 0 for fold in folds):
        raise argparse.ArgumentTypeError("folds must be a comma-separated list of nonnegative integers")
    return folds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_id", type=int, help="Prepared nnU-Net target dataset ID")
    parser.add_argument("--mode", choices=("scratch", "deepbranchai"), required=True)
    parser.add_argument("--folds", type=parse_folds, default=parse_folds("0,1,2,3,4"))
    parser.add_argument("--storage-dir", type=Path)
    parser.add_argument("--pretrained-checkpoint", type=Path)
    parser.add_argument("--source-fold", type=int, default=0, choices=range(5))
    parser.add_argument("--trainer", default="nnUNetTrainer_100epochs")
    parser.add_argument("--plans", default="nnUNetPlans")
    parser.add_argument("--configuration", default="3d_fullres")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--num-processes", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = setup_environment(storage_dir=args.storage_dir)
    checkpoint = None
    if args.mode == "deepbranchai":
        if args.pretrained_checkpoint:
            checkpoint = args.pretrained_checkpoint.resolve()
        else:
            checkpoint = ensure_pretrained_weights(
                FinetuneConfig(storage_dir=paths["storage"], pretrained_fold=args.source_fold)
            )

    print(f"Mode: {args.mode}")
    print(f"Target dataset: {args.dataset_id}")
    print(f"Folds: {args.folds}")
    if checkpoint:
        print(f"Initialization: {checkpoint}")

    for fold in args.folds:
        output = train_nnunet_fold(
            dataset_id=args.dataset_id,
            fold=fold,
            pretrained_weights=checkpoint,
            trainer=args.trainer,
            plans=args.plans,
            configuration=args.configuration,
            max_epochs=args.epochs,
            num_processes=args.num_processes,
            setup_env=False,
        )
        print(f"Completed fold {fold}: {output}")


if __name__ == "__main__":
    main()

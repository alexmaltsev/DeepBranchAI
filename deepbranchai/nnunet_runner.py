"""Small wrappers around nnU-Net commands and checkpoint layout."""

from __future__ import annotations

import functools
import shutil
import subprocess
import sys
from pathlib import Path

from .paths import setup_environment


def dataset_folder_name(dataset_id: int, dataset_name: str) -> str:
    safe_name = "".join(ch if ch.isalnum() else "_" for ch in dataset_name).strip("_")
    return f"Dataset{dataset_id:03d}_{safe_name}"


def run_command(cmd: list[str], cwd: str | Path | None = None) -> None:
    print(" ".join(str(part) for part in cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def run_planning_and_preprocessing(
    dataset_id: int,
    verify_dataset_integrity: bool = True,
    num_processes: int = 2,
) -> None:
    cmd = [
        sys.executable,
        "-c",
        (
            "from nnunetv2.experiment_planning.plan_and_preprocess_entrypoints "
            "import plan_and_preprocess_entry; plan_and_preprocess_entry()"
        ),
        "-d",
        str(dataset_id),
        "-np",
        str(num_processes),
    ]
    if verify_dataset_integrity:
        cmd.append("--verify_dataset_integrity")
    run_command(cmd)


def extract_fingerprints_only(
    dataset_id: int,
    verify_dataset_integrity: bool = True,
    num_processes: int = 2,
) -> None:
    from nnunetv2.experiment_planning.plan_and_preprocess_api import extract_fingerprints

    extract_fingerprints(
        [dataset_id],
        num_processes=num_processes,
        check_dataset_integrity=verify_dataset_integrity,
        clean=True,
        verbose=False,
    )


def preprocess_with_existing_plans(
    dataset_id: int,
    plans_identifier: str = "nnUNetPlans",
    configurations: tuple[str, ...] | list[str] = ("3d_fullres",),
    num_processes: int = 2,
) -> None:
    from nnunetv2.experiment_planning.plan_and_preprocess_api import preprocess

    config_list = list(configurations)
    process_list = [num_processes] * len(config_list)
    preprocess(
        [dataset_id],
        plans_identifier=plans_identifier,
        configurations=config_list,
        num_processes=process_list,
        verbose=False,
    )


def install_weights(
    extract_dir,
    nnunet_results,
    nnunet_preprocessed,
    nnunet_raw,
    dataset_name,
    trainer_dir,
    weight_subdir,
    config_prefix,
) -> None:
    """Install weights and configs into nnU-Net directory structure."""
    extract_dir = Path(extract_dir)
    nnunet_results = Path(nnunet_results)
    nnunet_preprocessed = Path(nnunet_preprocessed)
    nnunet_raw = Path(nnunet_raw)

    src_dir = extract_dir / weight_subdir
    for pth_file in sorted(src_dir.glob("*.pth")):
        fold = pth_file.stem.split("fold")[-1]
        dst_dir = nnunet_results / dataset_name / trainer_dir / f"fold_{fold}"
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / "checkpoint_best.pth"
        if not dst.exists():
            shutil.copy2(str(pth_file), str(dst))
            print(f"  Installed {pth_file.name} -> fold_{fold}/")
        else:
            print(f"  fold_{fold} already installed")

    trainer_dst_dir = nnunet_results / dataset_name / trainer_dir
    trainer_dst_dir.mkdir(parents=True, exist_ok=True)

    plans_src = extract_dir / "configs" / f"{config_prefix}_nnUNetPlans.json"
    if plans_src.exists():
        plans_preproc_dir = nnunet_preprocessed / dataset_name
        plans_preproc_dir.mkdir(parents=True, exist_ok=True)
        plans_preproc_dst = plans_preproc_dir / "nnUNetPlans.json"
        if not plans_preproc_dst.exists():
            shutil.copy2(str(plans_src), str(plans_preproc_dst))
            print("  Installed plans -> preprocessed/")
        plans_trainer_dst = trainer_dst_dir / "plans.json"
        if not plans_trainer_dst.exists():
            shutil.copy2(str(plans_src), str(plans_trainer_dst))
            print("  Installed plans -> results trainer dir")

    ds_src = extract_dir / "configs" / f"{config_prefix}_dataset.json"
    if ds_src.exists():
        ds_raw_dir = nnunet_raw / dataset_name
        ds_raw_dir.mkdir(parents=True, exist_ok=True)
        ds_raw_dst = ds_raw_dir / "dataset.json"
        if not ds_raw_dst.exists():
            shutil.copy2(str(ds_src), str(ds_raw_dst))
            print("  Installed dataset.json -> raw/")
        ds_trainer_dst = trainer_dst_dir / "dataset.json"
        if not ds_trainer_dst.exists():
            shutil.copy2(str(ds_src), str(ds_trainer_dst))
            print("  Installed dataset.json -> results trainer dir")

    print(f"  {config_prefix} setup complete.")


def finetune_from_pretrained(
    dataset_id: int,
    fold: int,
    pretrained_weights: str | Path,
    trainer: str = "nnUNetTrainer_100epochs",
    plans: str = "nnUNetPlans",
    configuration: str = "3d_fullres",
    max_epochs: int | None = None,
    num_processes: int = 4,
    setup_env: bool = True,
) -> Path:
    """Fine-tune a target dataset using nnU-Net's Python training API."""
    import torch

    if setup_env:
        setup_environment(verbose=False)
    pretrained_weights = Path(pretrained_weights)
    if not pretrained_weights.exists():
        raise FileNotFoundError(f"Pretrained weights not found: {pretrained_weights}")

    original_torch_load = torch.load

    @functools.wraps(original_torch_load)
    def patched_torch_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    torch.load = patched_torch_load
    try:
        from nnunetv2.run.run_training import get_trainer_from_args, maybe_load_checkpoint

        nnunet_trainer = get_trainer_from_args(
            str(dataset_id),
            configuration,
            fold,
            trainer,
            plans,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
        nnunet_trainer.num_processes = num_processes
        if max_epochs is not None:
            nnunet_trainer.num_epochs = max_epochs

        maybe_load_checkpoint(nnunet_trainer, False, False, str(pretrained_weights))
        nnunet_trainer.run_training()
        nnunet_trainer.perform_actual_validation(False)

        return Path(nnunet_trainer.output_folder) / "checkpoint_best.pth"
    finally:
        torch.load = original_torch_load


def predict_nnunet(
    input_dir: str | Path,
    output_dir: str | Path,
    dataset_id: int,
    fold: int,
    trainer: str = "nnUNetTrainer_100epochs",
    configuration: str = "3d_fullres",
    checkpoint_name: str = "checkpoint_best.pth",
    save_probabilities: bool = False,
) -> Path:
    """Run nnU-Net prediction on a prepared input directory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-c",
        "\n".join(
            [
                "import torch",
                "original_torch_load = torch.load",
                "def patched_torch_load(*args, **kwargs):",
                "    kwargs.setdefault('weights_only', False)",
                "    return original_torch_load(*args, **kwargs)",
                "torch.load = patched_torch_load",
                "from nnunetv2.inference.predict_from_raw_data import predict_entry_point",
                "predict_entry_point()",
            ]
        ),
        "-i",
        str(input_dir),
        "-o",
        str(output_dir),
        "-d",
        str(dataset_id),
        "-f",
        str(fold),
        "-tr",
        trainer,
        "-c",
        configuration,
        "-chk",
        checkpoint_name,
    ]
    if save_probabilities:
        cmd.append("--save_probabilities")
    run_command(cmd)
    return output_dir

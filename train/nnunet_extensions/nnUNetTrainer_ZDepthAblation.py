from copy import deepcopy

import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


def _plans_with_z_depth(plans: dict, configuration: str, z_depth: int) -> dict:
    patched = deepcopy(plans)
    cfg = patched["configurations"][configuration]
    original_patch_size = list(cfg["patch_size"])
    if original_patch_size != [352, 352, 128]:
        raise ValueError(
            f"Expected Dataset4005 paper patch size [352, 352, 128], got {original_patch_size}"
        )
    cfg["patch_size"] = [352, 352, z_depth]
    cfg["z_depth_ablation"] = {
        "baseline_patch_size": original_patch_size,
        "changed_axis": 2,
        "new_z_depth": z_depth,
        "controlled_variable": "patch_size[2]",
    }
    return patched


class nnUNetTrainer_100epochs_Z64(nnUNetTrainer):
    """Ordinary Dice/CE-Dice ablation with 352 x 352 x 64 patches."""

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(_plans_with_z_depth(plans, configuration, 64), configuration, fold, dataset_json, device)
        self.num_epochs = 100


class nnUNetTrainer_100epochs_Z32(nnUNetTrainer):
    """Ordinary Dice/CE-Dice ablation with 352 x 352 x 32 patches."""

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(_plans_with_z_depth(plans, configuration, 32), configuration, fold, dataset_json, device)
        self.num_epochs = 100

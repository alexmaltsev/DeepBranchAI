import torch

from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainer_CE_DC_CLDC import nnUNetTrainer_CE_DC_CLDC


class nnUNetTrainer_CE_DC_CLDC_100epochs(nnUNetTrainer_CE_DC_CLDC):
    """Paper-config clDice trainer: default nnU-Net behavior, CE + Dice + clDice loss, 100 epochs."""

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 100

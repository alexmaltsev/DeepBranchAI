"""Standalone training script for DeepBranchAI VESSEL12 fine-tuning."""
import os, sys, torch, functools
from multiprocessing import freeze_support

def main():
    sys.path.insert(0, os.path.abspath('..'))

    # Force single-threaded data augmentation to avoid worker OOM on Windows
    os.environ['nnUNet_n_proc_DA'] = '0'

    # Patch torch.load for PyTorch 2.6+ compatibility
    _original_torch_load = torch.load
    @functools.wraps(_original_torch_load)
    def _patched_torch_load(*args, **kwargs):
        kwargs.setdefault('weights_only', False)
        return _original_torch_load(*args, **kwargs)
    torch.load = _patched_torch_load

    from deepbranchai_utils import setup_environment

    BASE_DIR = 'F:/DeepBranchAI'
    paths = setup_environment(BASE_DIR)

    PRETRAINED_WEIGHTS = str(
        paths['nnUNet_results'] / 'Dataset4005_Mitochondria' /
        'nnUNetTrainer_100epochs__nnUNetPlans__3d_fullres' / 'fold_2' / 'checkpoint_best.pth'
    )
    FOLD = 2

    checkpoint = (
        paths['nnUNet_results'] / 'Dataset3005_Mitochondria' /
        'nnUNetTrainer_100epochs__nnUNetPlans__3d_fullres' /
        f'fold_{FOLD}' / 'checkpoint_best.pth'
    )

    if checkpoint.exists():
        print(f'Training already complete: {checkpoint}')
        return

    from nnunetv2.run.run_training import get_trainer_from_args, maybe_load_checkpoint
    import torch.backends.cudnn as cudnn

    print('Creating trainer...')
    nnunet_trainer = get_trainer_from_args(
        '3005', '3d_fullres', FOLD, 'nnUNetTrainer_100epochs', 'nnUNetPlans',
        device=torch.device('cuda'),
    )
    nnunet_trainer.num_processes = 4

    print('Loading pretrained weights...')
    maybe_load_checkpoint(nnunet_trainer, False, False, PRETRAINED_WEIGHTS)

    cudnn.deterministic = False
    cudnn.benchmark = True

    print('Starting training (100 epochs)...')
    nnunet_trainer.run_training()

    print('Running validation...')
    nnunet_trainer.perform_actual_validation(False)

    print('Done!')


if __name__ == '__main__':
    freeze_support()
    main()

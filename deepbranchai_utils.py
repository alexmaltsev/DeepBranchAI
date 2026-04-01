"""
DeepBranchAI utilities — nnU-Net setup, GPU checks, data downloads.

Import this at the top of any notebook:
    from deepbranchai_utils import setup_environment, download_and_extract, check_gpu
"""

import os
import sys
import zipfile
import shutil
import urllib.request
import json
from pathlib import Path


def setup_environment(base_dir):
    """
    Create nnU-Net directory structure and set environment variables.
    Works on Windows, Linux, and macOS.

    Returns a dict of all relevant paths.
    """
    base = Path(base_dir)
    paths = {
        'base':        base,
        'weights':     base / 'weights',
        'data':        base / 'data',
        'nnUNet_raw':  base / 'nnUNet_raw',
        'nnUNet_preprocessed': base / 'nnUNet_preprocessed',
        'nnUNet_results':      base / 'nnUNet_results',
    }

    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)

    # nnU-Net environment variables
    os.environ['nnUNet_raw']          = str(paths['nnUNet_raw'])
    os.environ['nnUNet_preprocessed'] = str(paths['nnUNet_preprocessed'])
    os.environ['nnUNet_results']      = str(paths['nnUNet_results'])

    # Fix MKL threading conflict (Linux with conda)
    os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

    print(f"Base directory:       {paths['base']}")
    print(f"nnUNet_raw:           {paths['nnUNet_raw']}")
    print(f"nnUNet_preprocessed:  {paths['nnUNet_preprocessed']}")
    print(f"nnUNet_results:       {paths['nnUNet_results']}")

    return paths


def check_gpu():
    """Verify CUDA is available and print GPU info."""
    import torch
    print(f"PyTorch version:  {torch.__version__}")
    print(f"CUDA available:   {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version:     {torch.version.cuda}")
        print(f"GPU:              {torch.cuda.get_device_name(0)}")
        print(f"VRAM:             {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("\n*** WARNING: No CUDA GPU detected. Training/inference will be extremely slow. ***")
        print("Install CUDA-enabled PyTorch:")
        print("  pip install torch --index-url https://download.pytorch.org/whl/cu118")
    return torch.cuda.is_available()


def download_and_extract(url, dest_dir, filename=None):
    """
    Download a file from URL and extract if it's a zip.
    Skips download if file already exists.

    Returns the path to the downloaded file.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        # Extract filename from URL
        filename = url.split('/')[-1].split('?')[0]

    filepath = dest_dir / filename

    if not filepath.exists():
        print(f"Downloading {filename} ...")
        urllib.request.urlretrieve(url, str(filepath))
        print(f"  Saved to {filepath}")
    else:
        print(f"  Already downloaded: {filepath}")

    # Extract zip files
    if filepath.suffix == '.zip':
        extract_dir = dest_dir / filepath.stem
        if not extract_dir.exists():
            print(f"  Extracting...")
            with zipfile.ZipFile(str(filepath), 'r') as zf:
                zf.extractall(str(dest_dir))
            print(f"  Extracted to {dest_dir}")
        else:
            print(f"  Already extracted: {extract_dir}")

    return filepath


def install_weights(extract_dir, nnunet_results, nnunet_preprocessed, nnunet_raw,
                    dataset_name, trainer_dir, weight_subdir, config_prefix):
    """
    Install weights + configs into nnU-Net directory structure.

    Parameters
    ----------
    extract_dir : Path to extracted Zenodo archive (contains weight dirs + configs/)
    dataset_name : e.g. 'Dataset3005_Mitochondria'
    trainer_dir : e.g. 'nnUNetTrainer_100epochs__nnUNetPlans__3d_fullres'
    weight_subdir : e.g. 'DeepBranchAI_VESSEL12_weights'
    config_prefix : e.g. 'DeepBranchAI_VESSEL12'
    """
    extract_dir = Path(extract_dir)
    nnunet_results = Path(nnunet_results)
    nnunet_preprocessed = Path(nnunet_preprocessed)
    nnunet_raw = Path(nnunet_raw)

    # Install weight files
    src_dir = extract_dir / weight_subdir
    for pth_file in sorted(src_dir.glob('*.pth')):
        fold = pth_file.stem.split('fold')[-1]
        dst_dir = nnunet_results / dataset_name / trainer_dir / f'fold_{fold}'
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / 'checkpoint_best.pth'
        if not dst.exists():
            shutil.copy2(str(pth_file), str(dst))
            print(f"  Installed {pth_file.name} -> fold_{fold}/")
        else:
            print(f"  fold_{fold} already installed")

    # Trainer output directory (where nnU-Net predictor looks for plans + dataset.json)
    trainer_dst_dir = nnunet_results / dataset_name / trainer_dir
    trainer_dst_dir.mkdir(parents=True, exist_ok=True)

    # Install plans
    plans_src = extract_dir / 'configs' / f'{config_prefix}_nnUNetPlans.json'
    if plans_src.exists():
        # nnU-Net preprocessed location
        plans_preproc_dir = nnunet_preprocessed / dataset_name
        plans_preproc_dir.mkdir(parents=True, exist_ok=True)
        plans_preproc_dst = plans_preproc_dir / 'nnUNetPlans.json'
        if not plans_preproc_dst.exists():
            shutil.copy2(str(plans_src), str(plans_preproc_dst))
            print(f"  Installed plans -> preprocessed/")
        # nnU-Net results trainer directory (required by predictor)
        plans_trainer_dst = trainer_dst_dir / 'plans.json'
        if not plans_trainer_dst.exists():
            shutil.copy2(str(plans_src), str(plans_trainer_dst))
            print(f"  Installed plans -> results trainer dir")

    # Install dataset.json
    ds_src = extract_dir / 'configs' / f'{config_prefix}_dataset.json'
    if ds_src.exists():
        # nnU-Net raw location
        ds_raw_dir = nnunet_raw / dataset_name
        ds_raw_dir.mkdir(parents=True, exist_ok=True)
        ds_raw_dst = ds_raw_dir / 'dataset.json'
        if not ds_raw_dst.exists():
            shutil.copy2(str(ds_src), str(ds_raw_dst))
            print(f"  Installed dataset.json -> raw/")
        # nnU-Net results trainer directory (required by predictor)
        ds_trainer_dst = trainer_dst_dir / 'dataset.json'
        if not ds_trainer_dst.exists():
            shutil.copy2(str(ds_src), str(ds_trainer_dst))
            print(f"  Installed dataset.json -> results trainer dir")

    print(f"  {config_prefix} setup complete.")

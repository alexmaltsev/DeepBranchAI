"""Download helpers."""

from __future__ import annotations

import shutil
import urllib.request
import zipfile
from pathlib import Path

from .nnunet_runner import install_weights


ZENODO_BASE = "https://zenodo.org/records/19363534/files"
ZENODO_PRETRAINED_WEIGHT = f"{ZENODO_BASE}/DeepBranchAI_MitoEye_fold2.pth?download=1"
ZENODO_PRETRAINED_PLANS = f"{ZENODO_BASE}/DeepBranchAI_MitoEye_nnUNetPlans.json?download=1"
ZENODO_PRETRAINED_DATASET = f"{ZENODO_BASE}/DeepBranchAI_MitoEye_dataset.json?download=1"
ZENODO_VESSEL12_WEIGHT = f"{ZENODO_BASE}/DeepBranchAI_VESSEL12_fold2.pth?download=1"
ZENODO_VESSEL12_PLANS = f"{ZENODO_BASE}/DeepBranchAI_VESSEL12_nnUNetPlans.json?download=1"
ZENODO_VESSEL12_DATASET = f"{ZENODO_BASE}/DeepBranchAI_VESSEL12_dataset.json?download=1"
ZENODO_VESSEL12_TRAINING = f"{ZENODO_BASE}/DeepBranchAI_VESSEL12_training.zip?download=1"
ZENODO_VESSEL12_DEMO = f"{ZENODO_BASE}/DeepBranchAI_demo_data.zip?download=1"


def download_file(url: str, dst: str | Path) -> Path:
    """Download ``url`` to ``dst`` unless it already exists."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        print(f"  Already downloaded: {dst}")
        return dst

    print(f"Downloading {dst.name} ...")
    urllib.request.urlretrieve(url, str(dst))
    print(f"  Saved to {dst}")
    return dst


def download_and_extract(url: str, dest_dir: str | Path, filename: str | None = None) -> Path:
    """Download a file and extract it when it is a zip archive."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = url.split("/")[-1].split("?")[0]

    filepath = download_file(url, dest_dir / filename)
    if filepath.suffix.lower() == ".zip":
        extract_dir = dest_dir / filepath.stem
        if extract_dir.exists():
            print(f"  Already extracted: {extract_dir}")
        else:
            print("  Extracting...")
            with zipfile.ZipFile(str(filepath), "r") as zf:
                zf.extractall(str(dest_dir))
            print(f"  Extracted to {dest_dir}")

    return filepath


def _safe_rmtree(path: str | Path, root: str | Path) -> None:
    path = Path(path).resolve()
    root = Path(root).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Refusing to remove {path} because it is outside {root}")
    shutil.rmtree(path)


def _download_named_files(files_to_download: list[tuple[str, Path]]) -> None:
    for url, dst in files_to_download:
        download_file(url, dst)


def download_and_install_pretrained_weights(paths: dict[str, Path]) -> Path:
    """Download the default pretrained checkpoint and install it into nnU-Net folders."""
    checkpoint = (
        paths["nnUNet_results"]
        / "Dataset4005_Mitochondria"
        / "nnUNetTrainer_100epochs__nnUNetPlans__3d_fullres"
        / "fold_2"
        / "checkpoint_best.pth"
    )
    if checkpoint.exists():
        print(f"Pretrained checkpoint ready: {checkpoint}")
        return checkpoint

    extract_dir = paths["weights"] / "DeepBranchAI_Zenodo"
    weight_dir = extract_dir / "DeepBranchAI_pretrained_weights"
    config_dir = extract_dir / "configs"
    weight_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)

    _download_named_files(
        [
            (ZENODO_PRETRAINED_WEIGHT, weight_dir / "DeepBranchAI_MitoEye_fold2.pth"),
            (ZENODO_PRETRAINED_PLANS, config_dir / "DeepBranchAI_MitoEye_nnUNetPlans.json"),
            (ZENODO_PRETRAINED_DATASET, config_dir / "DeepBranchAI_MitoEye_dataset.json"),
        ]
    )

    install_weights(
        extract_dir,
        paths["nnUNet_results"],
        paths["nnUNet_preprocessed"],
        paths["nnUNet_raw"],
        dataset_name="Dataset4005_Mitochondria",
        trainer_dir="nnUNetTrainer_100epochs__nnUNetPlans__3d_fullres",
        weight_subdir="DeepBranchAI_pretrained_weights",
        config_prefix="DeepBranchAI_MitoEye",
    )

    if not checkpoint.exists():
        raise FileNotFoundError(f"Pretrained checkpoint was not installed: {checkpoint}")
    print(f"Pretrained checkpoint ready: {checkpoint}")
    return checkpoint


def download_and_install_vessel12_weights(paths: dict[str, Path]) -> Path:
    """Download the VESSEL12 inference checkpoint and install it into nnU-Net folders."""
    checkpoint = (
        paths["nnUNet_results"]
        / "Dataset3005_Mitochondria"
        / "nnUNetTrainer_100epochs__nnUNetPlans__3d_fullres"
        / "fold_2"
        / "checkpoint_best.pth"
    )
    if checkpoint.exists():
        print(f"VESSEL12 checkpoint ready: {checkpoint}")
        return checkpoint

    extract_dir = paths["weights"] / "DeepBranchAI_Zenodo"
    weight_dir = extract_dir / "DeepBranchAI_VESSEL12_weights"
    config_dir = extract_dir / "configs"
    weight_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)

    _download_named_files(
        [
            (ZENODO_VESSEL12_WEIGHT, weight_dir / "DeepBranchAI_VESSEL12_fold2.pth"),
            (ZENODO_VESSEL12_PLANS, config_dir / "DeepBranchAI_VESSEL12_nnUNetPlans.json"),
            (ZENODO_VESSEL12_DATASET, config_dir / "DeepBranchAI_VESSEL12_dataset.json"),
        ]
    )

    install_weights(
        extract_dir,
        paths["nnUNet_results"],
        paths["nnUNet_preprocessed"],
        paths["nnUNet_raw"],
        dataset_name="Dataset3005_Mitochondria",
        trainer_dir="nnUNetTrainer_100epochs__nnUNetPlans__3d_fullres",
        weight_subdir="DeepBranchAI_VESSEL12_weights",
        config_prefix="DeepBranchAI_VESSEL12",
    )

    if not checkpoint.exists():
        raise FileNotFoundError(f"VESSEL12 checkpoint was not installed: {checkpoint}")
    print(f"VESSEL12 checkpoint ready: {checkpoint}")
    return checkpoint


def download_vessel12_reference_plans(paths: dict[str, Path]) -> Path:
    """Download the VESSEL12 nnU-Net plans file into the weights config folder."""
    target = paths["weights"] / "DeepBranchAI_Zenodo" / "configs" / "DeepBranchAI_VESSEL12_nnUNetPlans.json"
    download_file(ZENODO_VESSEL12_PLANS, target)
    return target


def find_vessel12_training_root(data_dir: str | Path) -> Path:
    """Find extracted VESSEL12 training data containing imagesTr/labelsTr."""
    data_dir = Path(data_dir)
    for candidate in (
        data_dir / "DeepBranchAI_VESSEL12_training",
        data_dir / "Dataset3005_Mitochondria",
        data_dir,
    ):
        if (candidate / "imagesTr").exists():
            return candidate

    for path in data_dir.rglob("imagesTr"):
        return path.parent

    raise FileNotFoundError("VESSEL12 training data not found after download/extract")


def download_vessel12_training_data(paths: dict[str, Path]) -> Path:
    """Download and extract the VESSEL12 training archive."""
    download_and_extract(
        ZENODO_VESSEL12_TRAINING,
        paths["data"],
        "DeepBranchAI_VESSEL12_training.zip",
    )
    training_root = find_vessel12_training_root(paths["data"])
    print(f"VESSEL12 training data ready: {training_root}")
    return training_root


def download_and_install_vessel12_training_data(
    paths: dict[str, Path],
    overwrite: bool = False,
) -> Path:
    """Download VESSEL12 training data and copy it into nnU-Net raw layout."""
    dataset_dir = paths["nnUNet_raw"] / "Dataset3005_Mitochondria"
    images_dir = dataset_dir / "imagesTr"
    labels_dir = dataset_dir / "labelsTr"
    if (
        not overwrite
        and images_dir.exists()
        and labels_dir.exists()
        and any(images_dir.glob("*.nii.gz"))
        and any(labels_dir.glob("*.nii.gz"))
    ):
        n_images = len(list(images_dir.glob("*.nii.gz")))
        n_labels = len(list(labels_dir.glob("*.nii.gz")))
        print(f"VESSEL12 training dataset installed: {n_images} images, {n_labels} labels")
        return dataset_dir

    training_root = download_vessel12_training_data(paths)
    if overwrite and dataset_dir.exists():
        _safe_rmtree(dataset_dir, paths["nnUNet_raw"])
    dataset_dir.mkdir(parents=True, exist_ok=True)

    for subdir in ("imagesTr", "labelsTr"):
        src = training_root / subdir
        dst = dataset_dir / subdir
        if not src.exists():
            print(f"WARNING: {src} not found")
            continue
        if overwrite and dst.exists():
            _safe_rmtree(dst, dataset_dir)
        if not dst.exists() or not any(dst.glob("*.nii.gz")):
            print(f"Copying {subdir}...")
            shutil.copytree(str(src), str(dst), dirs_exist_ok=True)
        else:
            existing = len(list(dst.glob("*.nii.gz")))
            print(f"{subdir} already installed ({existing} files)")

    ds_json_src = training_root / "dataset.json"
    ds_json_dst = dataset_dir / "dataset.json"
    if ds_json_src.exists() and (overwrite or not ds_json_dst.exists()):
        shutil.copy2(str(ds_json_src), str(ds_json_dst))

    n_images = len(list((dataset_dir / "imagesTr").glob("*.nii.gz")))
    n_labels = len(list((dataset_dir / "labelsTr").glob("*.nii.gz")))
    print(f"VESSEL12 training dataset installed: {n_images} images, {n_labels} labels")
    return dataset_dir


def download_vessel12_demo_data(paths: dict[str, Path]) -> Path:
    """Download and extract VESSEL12 demo/test volumes and annotations."""
    demo_dir = paths["data"] / "DeepBranchAI_demo_data"
    if demo_dir.exists() and any(demo_dir.rglob("*.tif")):
        print(f"VESSEL12 demo data ready: {demo_dir}")
        return demo_dir

    download_and_extract(
        ZENODO_VESSEL12_DEMO,
        paths["data"],
        "DeepBranchAI_demo_data.zip",
    )
    print(f"VESSEL12 demo data ready: {demo_dir}")
    return demo_dir


def find_vessel12_demo_volume(data_dir: str | Path, case_id: str = "VESSEL12_21") -> Path | None:
    """Find a demo raw volume that can be copied into the predict folder."""
    data_dir = Path(data_dir)
    preferred_names = [
        f"{case_id}_raw.tif",
        f"{case_id}_raw.tiff",
        f"{case_id}.nii.gz",
        f"{case_id}.nii",
        f"{case_id}.mha",
        f"{case_id}.mhd",
    ]
    for name in preferred_names:
        for path in data_dir.rglob(name):
            return path
    return None

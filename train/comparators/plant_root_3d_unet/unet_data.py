from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

STYLE_ROOT = Path(__file__).resolve().parent.parent / "soltaninejad_style"
sys.path.append(str(STYLE_ROOT))

from data import NpyVolume, TiffVolume, block_max_binary, block_mean, cached_paths, read_tiff_shape
from paths import CaseRecord


def parse_patch_size_zyx(value: int | str | tuple[int, int, int] | list[int]) -> tuple[int, int, int]:
    if isinstance(value, int):
        return (value, value, value)
    if isinstance(value, (tuple, list)):
        parts = [int(v) for v in value]
    else:
        text = str(value).lower().replace("x", ",").replace(";", ",").replace(" ", ",")
        parts = [int(item) for item in text.split(",") if item]
    if len(parts) != 3:
        raise ValueError(f"Expected one patch size or three ZYX sizes, got {value!r}")
    if any(part < 1 for part in parts):
        raise ValueError(f"Patch dimensions must be positive, got {parts}")
    return (parts[0], parts[1], parts[2])


class PlantRootUNetPatchDataset(Dataset):
    def __init__(
        self,
        records: list[CaseRecord],
        index_dir: Path,
        samples_per_epoch: int,
        patch_size_zyx: tuple[int, int, int],
        downsample_factor: int = 3,
        intensity_scale: float = 255.0,
        random_background_probability: float = 0.30,
        random_offset: int = 8,
        seed: int = 1337,
        cache_dir: Path | None = None,
    ):
        self.records = records
        self.index_dir = Path(index_dir)
        self.samples_per_epoch = int(samples_per_epoch)
        self.patch_size_zyx = tuple(int(v) for v in patch_size_zyx)
        self.downsample_factor = int(downsample_factor)
        self.intensity_scale = float(intensity_scale)
        self.random_background_probability = float(random_background_probability)
        self.random_offset = int(random_offset)
        self.seed = int(seed)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.use_cache = self.cache_dir is not None
        self._raw: dict[str, TiffVolume] = {}
        self._label: dict[str, TiffVolume] = {}
        self._cached_raw: dict[str, NpyVolume] = {}
        self._cached_label: dict[str, NpyVolume] = {}
        self._centers: dict[str, np.ndarray] = {}
        if self.use_cache:
            self._shapes = {}
            for record in records:
                raw_cache, label_cache = cached_paths(self.cache_dir, record)
                if not raw_cache.exists() or not label_cache.exists():
                    raise FileNotFoundError(f"Missing downsampled cache for {record.nnunet_id}: {raw_cache}, {label_cache}")
                self._shapes[record.nnunet_id] = tuple(int(v) for v in np.load(raw_cache, mmap_mode="r").shape)
        else:
            self._shapes = {record.nnunet_id: read_tiff_shape(record.raw) for record in records}

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _volume_pair(self, record: CaseRecord) -> tuple[TiffVolume, TiffVolume]:
        if record.nnunet_id not in self._raw:
            self._raw[record.nnunet_id] = TiffVolume(record.raw)
            self._label[record.nnunet_id] = TiffVolume(record.label)
        return self._raw[record.nnunet_id], self._label[record.nnunet_id]

    def _cached_volume_pair(self, record: CaseRecord) -> tuple[NpyVolume, NpyVolume]:
        if self.cache_dir is None:
            raise RuntimeError("cache_dir is not configured")
        if record.nnunet_id not in self._cached_raw:
            raw_cache, label_cache = cached_paths(self.cache_dir, record)
            self._cached_raw[record.nnunet_id] = NpyVolume(raw_cache)
            self._cached_label[record.nnunet_id] = NpyVolume(label_cache)
        return self._cached_raw[record.nnunet_id], self._cached_label[record.nnunet_id]

    def _load_centers(self, record: CaseRecord) -> np.ndarray:
        if record.nnunet_id not in self._centers:
            path = self.index_dir / f"{record.nnunet_id}_centers.npy"
            self._centers[record.nnunet_id] = np.load(path) if path.exists() else np.empty((0, 3), dtype=np.int32)
        return self._centers[record.nnunet_id]

    def _random_center(self, record: CaseRecord, rng: np.random.Generator) -> np.ndarray:
        shape = np.asarray(self._shapes[record.nnunet_id], dtype=np.int64)
        return rng.random(3) * np.maximum(shape - 1, 1)

    def _positive_center(self, record: CaseRecord, rng: np.random.Generator) -> np.ndarray | None:
        centers = self._load_centers(record)
        if centers.shape[0] == 0:
            return None
        center = centers[int(rng.integers(0, centers.shape[0]))].astype(np.float32)
        if self.use_cache:
            center = center / float(self.downsample_factor)
        jitter = rng.normal(0.0, self.random_offset * self.downsample_factor, size=3)
        if self.use_cache:
            jitter = jitter / float(self.downsample_factor)
        return center + jitter

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        rng = np.random.default_rng(self.seed + index)
        record = self.records[int(rng.integers(0, len(self.records)))]
        center = None
        if rng.random() >= self.random_background_probability:
            center = self._positive_center(record, rng)
        if center is None:
            center = self._random_center(record, rng)

        patch_size = np.asarray(self.patch_size_zyx, dtype=np.int64)
        if self.use_cache:
            raw_volume, label_volume = self._cached_volume_pair(record)
            start = np.rint(center - (patch_size / 2.0)).astype(np.int64)
            raw_patch = raw_volume.read_patch(tuple(int(v) for v in start), self.patch_size_zyx, fill_value=0)
            label_patch = label_volume.read_patch(tuple(int(v) for v in start), self.patch_size_zyx, fill_value=0)
            raw_patch = raw_patch.astype(np.float32, copy=False)
            label_patch = (label_patch > 0).astype(np.uint8, copy=False)
        else:
            raw_volume, label_volume = self._volume_pair(record)
            raw_size = patch_size * self.downsample_factor
            start = np.rint(center - (raw_size / 2.0)).astype(np.int64)
            raw_patch = raw_volume.read_patch(tuple(int(v) for v in start), tuple(int(v) for v in raw_size), fill_value=0)
            label_patch = label_volume.read_patch(tuple(int(v) for v in start), tuple(int(v) for v in raw_size), fill_value=0)
            raw_patch = block_mean(raw_patch, self.downsample_factor).astype(np.float32, copy=False)
            label_patch = block_max_binary(label_patch, self.downsample_factor)

        raw_patch = np.clip(raw_patch / self.intensity_scale, 0.0, 1.0)
        target = label_patch.astype(np.float32, copy=False)

        return {
            "input": torch.from_numpy(raw_patch[None, ...].copy()),
            "target": torch.from_numpy(target[None, ...].copy()),
            "case_id": record.nnunet_id,
            "source_id": record.source_id,
            "center_zyx": torch.as_tensor(np.rint(center).astype(np.int64)),
        }


def collate_unet_patch_samples(batch: list[dict]) -> dict[str, torch.Tensor | list[str]]:
    return {
        "input": torch.stack([item["input"] for item in batch]),
        "target": torch.stack([item["target"] for item in batch]),
        "case_id": [item["case_id"] for item in batch],
        "source_id": [item["source_id"] for item in batch],
        "center_zyx": torch.stack([item["center_zyx"] for item in batch]),
    }

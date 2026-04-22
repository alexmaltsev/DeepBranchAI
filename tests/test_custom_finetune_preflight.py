import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from deepbranchai.custom_finetune import (
    FinetuneConfig,
    inspect_custom_dataset,
    preflight_training_setup,
    prepare_nnunet_dataset,
)
from deepbranchai.image_io import save_tiff


def _make_raw(shape: tuple[int, int, int]) -> np.ndarray:
    arr = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape) % 255
    arr = arr + 1
    return arr


def _make_mask(shape: tuple[int, int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    z0, y0, x0 = [max(1, dim // 4) for dim in shape]
    z1, y1, x1 = [max(start + 1, (dim * 3) // 4) for start, dim in zip((z0, y0, x0), shape)]
    mask[z0:z1, y0:y1, x0:x1] = 1
    return mask


class CustomFinetunePreflightTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp(prefix="deepbranchai_preflight_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _storage_dir(self, name: str) -> Path:
        return self.tmpdir / name

    def _write_case(
        self,
        storage_dir: Path,
        case_id: str,
        raw_shape: tuple[int, int, int],
        mask_shape: tuple[int, int, int] | None = None,
    ) -> None:
        raw_dir = storage_dir / "data" / "custom_finetune" / "raw"
        gt_dir = storage_dir / "data" / "custom_finetune" / "ground_truth"
        raw_dir.mkdir(parents=True, exist_ok=True)
        gt_dir.mkdir(parents=True, exist_ok=True)
        save_tiff(_make_raw(raw_shape), raw_dir / f"{case_id}.tif")
        save_tiff(_make_mask(mask_shape or raw_shape), gt_dir / f"{case_id}_gt.tif")

    def _config(self, storage_dir: Path, dataset_id: int, project_name: str) -> FinetuneConfig:
        return FinetuneConfig(
            storage_dir=storage_dir,
            dataset_id=dataset_id,
            project_name=project_name,
            reference_patch_size=(128, 128, 64),
        )

    def test_single_case_partition_stays_within_partition_shape(self) -> None:
        storage_dir = self._storage_dir("single_case_valid")
        self._write_case(storage_dir, "case_00", (80, 64, 32))
        config = self._config(storage_dir, 9301, "single_case_valid")

        report = inspect_custom_dataset(config)
        with patch("deepbranchai.custom_finetune._gpu_info", return_value=(True, "Test GPU", 48.0)):
            preflight = preflight_training_setup(config, report=report, require_cuda=False)

        self.assertFalse(preflight.errors)
        self.assertEqual(preflight.split_mode, "single_volume_partition")
        self.assertEqual(preflight.train_source_case_ids, ["case_00"])
        self.assertEqual(preflight.validation_source_case_ids, ["case_00"])
        self.assertIsNotNone(preflight.train_shape_min)
        for patch_axis, shape_axis in zip(preflight.recommended_patch_size, preflight.train_shape_min):
            self.assertLessEqual(patch_axis, shape_axis)

        dataset_dir = prepare_nnunet_dataset(preflight.config, report=report)
        split = json.loads(
            (storage_dir / "nnUNet_preprocessed" / dataset_dir.name / "splits_final.json").read_text(encoding="utf-8")
        )[0]
        self.assertEqual(len(split["train"]), 1)
        self.assertEqual(len(split["val"]), 1)
        self.assertNotEqual(split["train"][0], split["val"][0])

    def test_single_case_with_too_small_axis_blocks_before_training(self) -> None:
        storage_dir = self._storage_dir("single_case_too_small")
        self._write_case(storage_dir, "case_00", (60, 48, 20))
        config = self._config(storage_dir, 9302, "single_case_too_small")

        report = inspect_custom_dataset(config)
        with patch("deepbranchai.custom_finetune._gpu_info", return_value=(True, "Test GPU", 48.0)):
            preflight = preflight_training_setup(config, report=report, require_cuda=False)

        self.assertTrue(preflight.errors)
        self.assertTrue(
            any("could not partition single volume" in message for message in preflight.errors),
            msg=preflight.errors,
        )

    def test_dimension_mismatch_is_caught(self) -> None:
        storage_dir = self._storage_dir("dimension_mismatch")
        self._write_case(storage_dir, "case_00", (80, 64, 32), mask_shape=(80, 64, 31))
        config = self._config(storage_dir, 9303, "dimension_mismatch")

        report = inspect_custom_dataset(config)
        self.assertTrue(report.errors)
        self.assertTrue(any("shape mismatch" in message for message in report.errors), msg=report.errors)

    def test_two_case_split_holds_out_one_case(self) -> None:
        storage_dir = self._storage_dir("two_cases")
        self._write_case(storage_dir, "case_00", (128, 128, 64))
        self._write_case(storage_dir, "case_01", (128, 128, 64))
        config = self._config(storage_dir, 9304, "two_cases")

        report = inspect_custom_dataset(config)
        with patch("deepbranchai.custom_finetune._gpu_info", return_value=(True, "Test GPU", 48.0)):
            preflight = preflight_training_setup(config, report=report, require_cuda=False)

        self.assertFalse(preflight.errors)
        self.assertEqual(preflight.split_mode, "auto_single_validation")
        self.assertEqual(len(preflight.train_source_case_ids), 1)
        self.assertEqual(len(preflight.validation_source_case_ids), 1)

    def test_six_case_split_uses_four_fifths_and_24gb_patch_reduction(self) -> None:
        storage_dir = self._storage_dir("six_cases")
        for i in range(6):
            self._write_case(storage_dir, f"case_{i:02d}", (128, 128, 64))
        config = self._config(storage_dir, 9305, "six_cases")

        report = inspect_custom_dataset(config)
        with patch("deepbranchai.custom_finetune._gpu_info", return_value=(True, "Test GPU", 24.0)):
            preflight = preflight_training_setup(config, report=report, require_cuda=False)

        self.assertFalse(preflight.errors)
        self.assertEqual(preflight.split_mode, "auto_four_fifths")
        self.assertEqual(len(preflight.train_source_case_ids), 4)
        self.assertEqual(len(preflight.validation_source_case_ids), 2)
        self.assertEqual(preflight.recommended_patch_size, (96, 96, 48))
        self.assertIsNotNone(preflight.train_shape_min)
        for patch_axis, shape_axis in zip(preflight.recommended_patch_size, preflight.train_shape_min):
            self.assertLessEqual(patch_axis, shape_axis)


if __name__ == "__main__":
    unittest.main()

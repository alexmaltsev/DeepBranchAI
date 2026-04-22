import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile

from deepbranchai.custom_finetune import FinetuneConfig, import_user_input_folders, resolve_dataset_id_conflict
from deepbranchai.image_io import load_volume, save_tiff


class CustomFinetuneInputImportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp(prefix="deepbranchai_inputs_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _config(self) -> FinetuneConfig:
        return FinetuneConfig(storage_dir=self.tmpdir / "workspace", dataset_id=9401, project_name="input_import")

    def test_import_user_input_folders_converts_rgb_raw_and_binarizes_mask(self) -> None:
        raw_dir = self.tmpdir / "input_raw"
        gt_dir = self.tmpdir / "input_gt"
        infer_dir = self.tmpdir / "input_infer"
        raw_dir.mkdir(parents=True)
        gt_dir.mkdir(parents=True)
        infer_dir.mkdir(parents=True)

        rgb = np.zeros((8, 16, 16, 3), dtype=np.uint8)
        rgb[..., 0] = 10
        rgb[..., 1] = 20
        rgb[..., 2] = 30
        tifffile.imwrite(raw_dir / "sample.tif", rgb)

        mask = np.zeros((8, 16, 16), dtype=np.uint8)
        mask[:, 4:12, 4:12] = 127
        mask[:, 6:10, 6:10] = 255
        save_tiff(mask, gt_dir / "sample_mask.tif")

        infer_rgb = np.zeros((8, 16, 16, 3), dtype=np.uint8)
        infer_rgb[..., 0] = 5
        infer_rgb[..., 1] = 15
        infer_rgb[..., 2] = 25
        tifffile.imwrite(infer_dir / "to_segment.tif", infer_rgb)

        report = import_user_input_folders(
            self._config(),
            training_raw_input_dir=raw_dir,
            training_ground_truth_input_dir=gt_dir,
            inference_input_dir=infer_dir,
        )

        self.assertFalse(report.errors, msg=report.errors)
        self.assertEqual(len(report.staged_pairs), 1)
        self.assertEqual(len(report.staged_predict_inputs), 1)

        staged_pair = report.staged_pairs[0]
        self.assertTrue(staged_pair.raw_path.name.endswith(".tif"))
        self.assertEqual(staged_pair.mask_path.name, "sample_gt.tif")

        raw = load_volume(staged_pair.raw_path)
        gt = load_volume(staged_pair.mask_path)
        pred_input = load_volume(report.staged_predict_inputs[0])

        self.assertEqual(raw.shape, (8, 16, 16))
        self.assertEqual(pred_input.shape, (8, 16, 16))
        self.assertEqual(gt.shape, (8, 16, 16))
        self.assertSetEqual(set(np.unique(gt).tolist()), {0, 1})
        self.assertTrue(any("grayscale" in warning.lower() for warning in report.warnings))
        self.assertTrue(any("foreground > 0" in warning.lower() for warning in report.warnings))

    def test_import_user_input_folders_uses_raw_folder_for_inference_when_not_set(self) -> None:
        raw_dir = self.tmpdir / "input_raw"
        gt_dir = self.tmpdir / "input_gt"
        raw_dir.mkdir(parents=True)
        gt_dir.mkdir(parents=True)

        raw = np.arange(8 * 12 * 12, dtype=np.uint16).reshape((8, 12, 12)) + 1
        mask = np.zeros((8, 12, 12), dtype=np.uint8)
        mask[:, 2:10, 2:10] = 1
        save_tiff(raw, raw_dir / "case_a.tif")
        save_tiff(mask, gt_dir / "case_a.tif")

        report = import_user_input_folders(
            self._config(),
            training_raw_input_dir=raw_dir,
            training_ground_truth_input_dir=gt_dir,
            inference_input_dir=None,
        )

        self.assertFalse(report.errors, msg=report.errors)
        self.assertEqual(len(report.staged_predict_inputs), 1)
        self.assertTrue(any("used for inference" in warning.lower() for warning in report.warnings))

    def test_resolve_dataset_id_conflict_moves_to_next_free_id(self) -> None:
        workspace = self.tmpdir / "workspace"
        for root_name in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
            (workspace / root_name / "Dataset9401_existing_demo").mkdir(parents=True, exist_ok=True)

        config = self._config()
        adjusted, note = resolve_dataset_id_conflict(config)

        self.assertEqual(adjusted.dataset_id, 9402)
        self.assertIsNotNone(note)
        self.assertIn("9401", note)
        self.assertIn("9402", note)


if __name__ == "__main__":
    unittest.main()

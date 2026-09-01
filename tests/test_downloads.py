import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from deepbranchai.downloads import (
    download_and_install_pretrained_weights,
    pretrained_weight_url,
)


class DownloadTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="deepbranchai_downloads_"))
        self.paths = {
            name: self.root / name
            for name in ("weights", "nnUNet_results", "nnUNet_preprocessed", "nnUNet_raw")
        }
        for path in self.paths.values():
            path.mkdir(parents=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def test_urls_cover_all_released_folds(self) -> None:
        for fold in range(5):
            self.assertIn(f"DeepBranchAI_MitoEye_fold{fold}.pth", pretrained_weight_url(fold))
        with self.assertRaises(ValueError):
            pretrained_weight_url(5)

    def test_installs_selected_fold(self) -> None:
        checkpoint = (
            self.paths["nnUNet_results"]
            / "Dataset4005_Mitochondria"
            / "nnUNetTrainer_100epochs__nnUNetPlans__3d_fullres"
            / "fold_3"
            / "checkpoint_best.pth"
        )

        def fake_install(*args, **kwargs):
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_bytes(b"test")

        with (
            patch("deepbranchai.downloads._download_named_files") as download,
            patch("deepbranchai.downloads.install_weights", side_effect=fake_install),
        ):
            installed = download_and_install_pretrained_weights(self.paths, fold=3)

        self.assertEqual(installed, checkpoint)
        requested = download.call_args.args[0]
        self.assertIn("DeepBranchAI_MitoEye_fold3.pth", requested[0][0])


if __name__ == "__main__":
    unittest.main()

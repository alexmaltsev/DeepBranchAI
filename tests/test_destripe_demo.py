import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile

from deepbranchai.destripe_demo import DestripeDemoConfig, prepare_destripe_demo, run_destripe


class DestripeDemoTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp(prefix="deepbranchai_destripe_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_prepare_destripe_demo_converts_rgb_input_to_grayscale(self) -> None:
        rgb = np.zeros((24, 20, 3), dtype=np.uint8)
        rgb[..., 0] = 10
        rgb[..., 1] = 30
        rgb[..., 2] = 90
        input_path = self.tmpdir / "rgb_input.tif"
        tifffile.imwrite(str(input_path), rgb)

        config = DestripeDemoConfig(storage_dir=self.tmpdir / "workspace")
        state = prepare_destripe_demo(config, input_path=input_path)

        self.assertFalse(state.used_synthetic_demo)
        self.assertEqual(state.input_array.shape, (24, 20))
        self.assertTrue(any("grayscale" in warning.lower() for warning in state.warnings))
        self.assertTrue(state.input_path.exists())

    def test_run_destripe_accepts_three_dimensional_volume(self) -> None:
        volume = np.stack([np.full((16, 16), idx, dtype=np.float32) for idx in range(6)], axis=0)
        input_path = self.tmpdir / "volume.tif"
        tifffile.imwrite(str(input_path), volume)

        config = DestripeDemoConfig(storage_dir=self.tmpdir / "workspace")
        state = prepare_destripe_demo(config, input_path=input_path)

        def identity_filter(image, **kwargs):
            return image

        output_path = run_destripe(state, filter_fn=identity_filter)
        self.assertTrue(output_path.exists())
        self.assertEqual(state.filtered_array.shape, volume.shape)


if __name__ == "__main__":
    unittest.main()

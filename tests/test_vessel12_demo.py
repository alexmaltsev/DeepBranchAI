import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from deepbranchai.vessel12_demo import Vessel12DemoConfig, Vessel12DemoState, compute_annotation_metrics, load_vessel12_prediction


class Vessel12DemoTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp(prefix="deepbranchai_vessel12_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_load_vessel12_prediction_reads_probability_npz(self) -> None:
        output_dir = self.tmpdir / "predictions"
        output_dir.mkdir(parents=True)
        logits = np.zeros((2, 4, 6, 8), dtype=np.float32)
        logits[1, 1:3, 2:5, 3:7] = 0.9
        np.savez(output_dir / "case.npz", probabilities=logits)

        prob_map, mask = load_vessel12_prediction(output_dir, expected_shape=(4, 6, 8), threshold=0.5)

        self.assertEqual(prob_map.shape, (4, 6, 8))
        self.assertEqual(mask.shape, (4, 6, 8))
        self.assertEqual(int(mask.sum()), 24)

    def test_compute_annotation_metrics_uses_scaled_coordinates_when_needed(self) -> None:
        annotation_path = self.tmpdir / "annotations.csv"
        pd.DataFrame(
            [
                [1, 1, 1, 1],
                [0, 0, 0, 0],
            ]
        ).to_csv(annotation_path, header=False, index=False)

        mask = np.zeros((6, 6, 6), dtype=np.uint8)
        mask[2, 2, 2] = 1

        state = Vessel12DemoState(
            config=Vessel12DemoConfig(storage_dir=self.tmpdir / "workspace"),
            used_demo_data=False,
            input_path=self.tmpdir / "input.tif",
            annotation_path=annotation_path,
            output_dir=self.tmpdir / "predictions",
            volume=np.zeros((6, 6, 6), dtype=np.float32),
            mask=mask,
        )

        metrics = compute_annotation_metrics(state)

        self.assertAlmostEqual(metrics["accuracy"], 1.0)
        self.assertEqual(state.annotation_scale_note, "2x scaled coordinates")


if __name__ == "__main__":
    unittest.main()

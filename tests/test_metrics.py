import unittest

import numpy as np

from deepbranchai.metrics import cldice_score, compute_binary_volume_metrics


class VolumeMetricTests(unittest.TestCase):
    def test_identical_branching_masks_are_perfect(self) -> None:
        reference = np.zeros((7, 7, 7), dtype=np.uint8)
        reference[3, 1:6, 3] = 1
        reference[3, 3, 1:6] = 1

        metrics = compute_binary_volume_metrics(reference, reference)

        self.assertEqual(metrics["dice"], 1.0)
        self.assertEqual(metrics["cldice"], 1.0)
        self.assertEqual(metrics["abs_cc_error"], 0)
        self.assertEqual(metrics["avd_percent"], 0.0)

    def test_broken_branch_reduces_cldice_and_changes_component_count(self) -> None:
        reference = np.zeros((9, 9, 9), dtype=np.uint8)
        reference[4, 1:8, 4] = 1
        prediction = reference.copy()
        prediction[4, 4, 4] = 0

        metrics = compute_binary_volume_metrics(prediction, reference)

        self.assertLess(metrics["cldice"], 1.0)
        self.assertEqual(metrics["abs_cc_error"], 1)

    def test_cldice_handles_empty_masks(self) -> None:
        empty = np.zeros((3, 3, 3), dtype=np.uint8)
        foreground = empty.copy()
        foreground[1, 1, 1] = 1

        self.assertEqual(cldice_score(empty, empty), 1.0)
        self.assertEqual(cldice_score(empty, foreground), 0.0)

    def test_rejects_non_volume_input(self) -> None:
        with self.assertRaises(ValueError):
            compute_binary_volume_metrics(np.zeros((3, 3)), np.zeros((3, 3)))


if __name__ == "__main__":
    unittest.main()

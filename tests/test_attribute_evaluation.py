import unittest

import torch

from transkun.AttributeEvaluation import (
    average_precision,
    expected_calibration_error,
    summarize_reference_attributes,
)


class AttributeEvaluationTest(unittest.TestCase):
    def test_average_precision_perfect_ranking(self):
        self.assertEqual(average_precision([0.9, 0.8, 0.1], [1, 1, 0]), 1.0)

    def test_expected_calibration_error(self):
        value = expected_calibration_error([0.9, 0.6], [1, 0], n_bins=2)
        self.assertAlmostEqual(value, 0.25)

    def test_structured_summary_masks_missing_boundaries(self):
        batch = {
            "parameterization": "categorical",
            "velocity": torch.tensor([60, 80]),
            "timing": torch.tensor([[0.0, 0.5], [-0.5, 0.0]]),
            "presence": torch.tensor([[True, False], [False, True]]),
            "pitch": torch.tensor([60, 72]),
            "duration": torch.tensor([0.2, 1.2]),
            "outputs": {
                "velocity": torch.nn.functional.one_hot(
                    torch.tensor([60, 80]), 128).float() * 10,
                "onsetTiming": torch.tensor([[0., 0., 10.], [10., 0., 0.]]),
                "offsetTiming": torch.tensor([[0., 0., 10.], [0., 10., 0.]]),
                "onsetPresence": torch.tensor([[0., 10.], [10., 0.]]),
                "offsetPresence": torch.tensor([[10., 0.], [0., 10.]]),
            },
        }
        summary = summarize_reference_attributes(batch, hop_seconds=0.02)

        self.assertEqual(summary["onset_timing"]["count"], 1)
        self.assertEqual(summary["offset_timing"]["count"], 1)
        self.assertEqual(summary["velocity"]["accuracy"], 1.0)
        self.assertEqual(summary["onset_presence"]["f1"], 1.0)
        self.assertAlmostEqual(
            summary["offset_timing"]["mean_absolute_error_seconds"], 0.0)

    def test_timing_error_uses_continuous_reference_not_bin_center(self):
        batch = {
            "parameterization": "categorical",
            "velocity": torch.tensor([60]),
            "timing": torch.tensor([[0.1, 0.0]]),
            "presence": torch.tensor([[True, True]]),
            "pitch": torch.tensor([60]),
            "duration": torch.tensor([0.5]),
            "outputs": {
                "velocity": torch.nn.functional.one_hot(
                    torch.tensor([60]), 128).float() * 10,
                "onsetTiming": torch.tensor([[0., 10., 0.]]),
                "offsetTiming": torch.tensor([[0., 10., 0.]]),
                "onsetPresence": torch.tensor([[0., 10.]]),
                "offsetPresence": torch.tensor([[0., 10.]]),
            },
        }
        summary = summarize_reference_attributes(batch, hop_seconds=0.02)

        self.assertAlmostEqual(
            summary["onset_timing"]["mean_absolute_error"], 0.1,
            places=5)
        self.assertAlmostEqual(
            summary["onset_timing"]["mean_absolute_error_seconds"],
            0.002, places=5)

    def test_legacy_summary_masks_missing_timing_boundaries(self):
        batch = {
            "parameterization": "continuous_bernoulli",
            "velocity": torch.tensor([60, 70]),
            "timing": torch.tensor([[0.0, 0.2], [-0.2, 0.0]]),
            "presence": torch.tensor([[True, False], [False, True]]),
            "pitch": torch.tensor([60, 70]),
            "duration": torch.tensor([0.4, 0.6]),
            "outputs": {
                "velocity": torch.zeros(2, 128),
                "timing": torch.zeros(2, 2),
                "presence": torch.tensor([[10., -10.], [-10., 10.]]),
            },
        }
        summary = summarize_reference_attributes(batch, hop_seconds=0.02)

        self.assertEqual(summary["onset_timing"]["count"], 1)
        self.assertEqual(summary["offset_timing"]["count"], 1)
        self.assertEqual(summary["onset_presence"]["f1"], 1.0)
        self.assertEqual(summary["offset_presence"]["f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
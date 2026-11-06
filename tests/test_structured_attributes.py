import unittest

import torch

from transkun import CRF
from transkun.StructuredAttributes import (
    add_attribute_scores,
    attribute_log_prob,
    gather_attribute_logits,
    timing_index_to_value,
    timing_value_to_index,
)


def _attribute_logits(time_steps):
    return {
        "velocity": torch.zeros(1, 1, time_steps, 128, requires_grad=True),
        "onsetTiming": torch.zeros(1, 1, time_steps, 5, requires_grad=True),
        "offsetTiming": torch.zeros(1, 1, time_steps, 5, requires_grad=True),
        "onsetPresence": torch.zeros(1, 1, time_steps, 2, requires_grad=True),
        "offsetPresence": torch.zeros(1, 1, time_steps, 2, requires_grad=True),
    }


class StructuredAttributeTest(unittest.TestCase):
    def test_attribute_evidence_changes_interval_decoding(self):
        score = torch.full((3, 3, 1), -20.0)
        score[1, 0, 0] = 0.0
        score[2, 0, 0] = 0.1
        crf = CRF.NeuralSemiCRFInterval(score, torch.zeros(2, 1))
        self.assertEqual(crf.decode(), [[(0, 2)]])

        attributes = _attribute_logits(3)
        attributes["offsetTiming"] = attributes["offsetTiming"].detach()
        attributes["offsetTiming"][0, 0, 1, 0] = 3.0
        joint = add_attribute_scores(crf, attributes, weight=1.0)
        self.assertEqual(joint.decode(), [[(0, 1)]])

    def test_joint_label_score_backpropagates_to_all_attributes(self):
        attributes = _attribute_logits(3)
        intervals = [[[(0, 2)]]]
        gathered, scatter = gather_attribute_logits(attributes, intervals, n_symbols=1)
        score = attribute_log_prob(
            gathered, [64], [(0.0, 0.25)], [(True, False)],
            n_timing_bins=5, weight=1.0)
        self.assertEqual(scatter.tolist(), [0])
        (-score.sum()).backward()
        for logits in attributes.values():
            self.assertIsNotNone(logits.grad)
            self.assertGreater(logits.grad.abs().sum().item(), 0)

    def test_crf_partition_backpropagates_attribute_evidence(self):
        attributes = _attribute_logits(3)
        crf = CRF.NeuralSemiCRFInterval(
            torch.zeros(3, 3, 1), torch.zeros(2, 1))
        joint = add_attribute_scores(crf, attributes, weight=1.0)
        loss = -joint.logProb([[(0, 2)]]).sum()
        loss.backward()
        self.assertGreater(attributes["velocity"].grad.abs().sum().item(), 0)
        self.assertGreater(attributes["offsetTiming"].grad.abs().sum().item(), 0)

    def test_timing_quantization_round_trip(self):
        values = torch.tensor([-0.5, -0.26, 0.0, 0.24, 0.5])
        indices = timing_value_to_index(values, n_timing_bins=5)
        reconstructed = timing_index_to_value(indices, n_timing_bins=5)
        self.assertTrue(torch.all((values-reconstructed).abs() <= 0.125))


if __name__ == "__main__":
    unittest.main()
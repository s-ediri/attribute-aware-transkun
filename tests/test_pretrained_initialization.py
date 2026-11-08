import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from transkun.TrainUtil import load_pretrained_weights


class SmallModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(2, 2)
        self.scorer = nn.Linear(2, 1)
        self.structuredAttributePredictor = nn.Linear(2, 3)


class PretrainedInitializationTest(unittest.TestCase):
    def test_loads_compatible_weights_and_keeps_new_head_initialized(self):
        model = SmallModel()
        initial_head = {
                key: value.clone()
                for key, value in model.structuredAttributePredictor.state_dict().items()
                }
        source = {
                "backbone.weight": torch.full_like(model.backbone.weight, 2),
                "backbone.bias": torch.full_like(model.backbone.bias, 2),
                "scorer.weight": torch.full_like(model.scorer.weight, 3),
                "scorer.bias": torch.full_like(model.scorer.bias, 3),
                "velocityPredictor.weight": torch.ones(1),
                }

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "pretrained.pt"
            torch.save({"state_dict": source}, checkpoint_path)
            with contextlib.redirect_stdout(io.StringIO()):
                load_pretrained_weights(model, checkpoint_path, "cpu")

        self.assertTrue(torch.all(model.backbone.weight == 2))
        self.assertTrue(torch.all(model.scorer.weight == 3))
        for key, value in model.structuredAttributePredictor.state_dict().items():
            self.assertTrue(torch.equal(value, initial_head[key]))

    def test_rejects_checkpoint_without_core_model_weights(self):
        model = SmallModel()
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "incompatible.pt"
            torch.save({"state_dict": {"velocityPredictor.weight": torch.ones(1)}},
                       checkpoint_path)

            with self.assertRaisesRegex(ValueError, "core model weights"):
                load_pretrained_weights(model, checkpoint_path, "cpu")

    def test_strict_loading_rejects_missing_or_extra_tensors(self):
        model = SmallModel()
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "partial.pt"
            torch.save({"state_dict": {"backbone.weight": model.backbone.weight}},
                       checkpoint_path)

            with self.assertRaises(RuntimeError):
                load_pretrained_weights(
                    model, checkpoint_path, "cpu", strict=True)


if __name__ == "__main__":
    unittest.main()

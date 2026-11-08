import json
import unittest
from pathlib import Path

import torch

from transkun import ModelTransformerLegacy


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class LegacyControlTest(unittest.TestCase):
    def test_packaged_checkpoint_matches_legacy_architecture_strictly(self):
        configuration = json.loads(
            (REPOSITORY_ROOT / "experiments/configs/legacy_finetune.conf").read_text(
                encoding="utf-8"))["Model"]["config"]
        config = ModelTransformerLegacy.Config()
        for key, value in configuration.items():
            setattr(config, key, value)

        model = ModelTransformerLegacy.TransKun(config)
        checkpoint = torch.load(
            REPOSITORY_ROOT / "transkun/pretrained/2.0.pt",
            map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["state_dict"], strict=True)

        self.assertEqual(
            ModelTransformerLegacy.SOURCE_COMMIT,
            "03b23776bae3c8b3046dcbe1f2b17bea64c5b25f")
        self.assertTrue(hasattr(model, "velocityPredictor"))
        self.assertTrue(hasattr(model, "refinedOFPredictor"))
        self.assertFalse(hasattr(model, "structuredAttributePredictor"))


if __name__ == "__main__":
    unittest.main()
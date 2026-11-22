import json
import unittest
from pathlib import Path

import torch

from transkun import ModelTransformerLegacy
from transkun import ModelTransformer
from transkun.TrainUtil import load_pretrained_weights


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
            REPOSITORY_ROOT / "transkun/pretrained/Transkun V2.pt",
            map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["state_dict"], strict=True)

        self.assertEqual(
            ModelTransformerLegacy.SOURCE_COMMIT,
            "03b23776bae3c8b3046dcbe1f2b17bea64c5b25f")
        self.assertTrue(hasattr(model, "velocityPredictor"))
        self.assertTrue(hasattr(model, "refinedOFPredictor"))
        self.assertFalse(hasattr(model, "structuredAttributePredictor"))

    def test_structured_controls_load_corrected_baseline_compatibly(self):
        checkpoint = REPOSITORY_ROOT / "transkun/pretrained/Transkun V2.pt"
        for config_name in ("posthoc_attributes.conf", "joint_attributes.conf"):
            configuration = json.loads((
                REPOSITORY_ROOT / "experiments/configs" / config_name
            ).read_text(encoding="utf-8"))["Model"]["config"]
            config = ModelTransformer.Config()
            for key, value in configuration.items():
                setattr(config, key, value)
            model = ModelTransformer.TransKun(config)
            load_pretrained_weights(model, checkpoint, "cpu", strict=False)

            state = torch.load(
                checkpoint, map_location="cpu", weights_only=False)["state_dict"]
            self.assertTrue(torch.equal(
                model.backbone.posEmbedBuilder.proj.weight,
                state["backbone.posEmbedBuilder.proj.weight"]))
            self.assertTrue(hasattr(model, "structuredAttributePredictor"))


if __name__ == "__main__":
    unittest.main()
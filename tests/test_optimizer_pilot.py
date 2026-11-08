import copy
import random
import unittest

import numpy as np
import torch

from scripts.optimizer_pilot import (
    capture_rng_state,
    restore_rng_state,
    validate_resume_metadata,
)


class OptimizerPilotTest(unittest.TestCase):
    def metadata(self):
        return {
            "workflow": "bounded_optimizer_pilot",
            "system_id": "C",
            "seed": 123,
            "model_module": "module",
            "model_config": {"size": 1},
            "training": {"optimizer_steps": 10000},
            "dataset_root": "dataset",
            "files": {"model_config": {"sha256": "abc"}},
        }

    def test_resume_accepts_identical_metadata(self):
        metadata = self.metadata()
        self.assertIs(validate_resume_metadata(metadata, copy.deepcopy(metadata)),
                      metadata)

    def test_resume_rejects_changed_system(self):
        saved = self.metadata()
        proposed = copy.deepcopy(saved)
        proposed["system_id"] = "D"
        with self.assertRaisesRegex(ValueError, "system_id"):
            validate_resume_metadata(saved, proposed)

    def test_resume_rejects_changed_schedule(self):
        saved = self.metadata()
        proposed = copy.deepcopy(saved)
        proposed["training"]["optimizer_steps"] = 20
        with self.assertRaisesRegex(ValueError, "training"):
            validate_resume_metadata(saved, proposed)

    def test_rng_state_round_trip(self):
        random.seed(1)
        np.random.seed(1)
        torch.manual_seed(1)
        state = capture_rng_state()
        expected = (random.random(), np.random.rand(), torch.rand(1).item())
        restore_rng_state(state)
        actual = (random.random(), np.random.rand(), torch.rand(1).item())
        self.assertEqual(expected, actual)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_rng_restore_accepts_device_mapped_torch_state(self):
        state = capture_rng_state()
        state["torch"] = state["torch"].cuda()
        state["cuda"] = [item.cuda() for item in state["cuda"]]
        restore_rng_state(state)


if __name__ == "__main__":
    unittest.main()

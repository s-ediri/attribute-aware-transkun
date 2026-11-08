import copy
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
from torch import nn

from scripts.optimizer_pilot import (
    capture_rng_state,
    load_or_initialize,
    restore_rng_state,
    validate_resume_metadata,
)
from transkun.TrainUtil import save_checkpoint


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

    def create_components(self):
        model = nn.Linear(2, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=1e-3, total_steps=3)
        return model, optimizer, scheduler

    def test_resume_loader_preserves_loss_history(self):
        model, optimizer, scheduler = self.create_components()
        history = {"train": [3.0, 2.0], "val": []}
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "checkpoint.pt"
            save_checkpoint(
                path, epoch=0, nIter=2, model=model,
                lossTracker=history, best_state_dict=model.state_dict(),
                optimizer=optimizer, lrScheduler=scheduler,
                metadata=self.metadata())
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            checkpoint["rng_state"] = capture_rng_state()
            torch.save(checkpoint, path)
            new_model, new_optimizer, new_scheduler = self.create_components()
            initialized = (
                0, 0, new_model, {"train": [], "val": []},
                new_model.state_dict(), new_optimizer, new_scheduler)
            args = SimpleNamespace(
                checkpoint=path, max_lr=1e-3, weight_decay=1e-4,
                schedule_steps=3, pretrained_checkpoint=None,
                strict_pretrained=False)
            with mock.patch(
                    "scripts.optimizer_pilot.initializeCheckpoint",
                    return_value=initialized):
                loaded = load_or_initialize(
                    args, nn.Linear, object(), torch.device("cpu"),
                    self.metadata())
        self.assertEqual(loaded[3], history)


if __name__ == "__main__":
    unittest.main()

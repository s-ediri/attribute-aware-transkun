import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from transkun.TrainUtil import save_checkpoint
from transkun.train import resolve_run_metadata


class CheckpointMetadataTest(unittest.TestCase):
    def create_checkpoint(self, path, metadata):
        model = nn.Linear(2, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=1e-3, total_steps=2)
        save_checkpoint(
            path, epoch=0, nIter=0, model=model,
            lossTracker={"train": [], "val": []},
            best_state_dict=model.state_dict(), optimizer=optimizer,
            lrScheduler=scheduler, metadata=metadata)

    def test_metadata_survives_checkpoint_save_and_resume(self):
        metadata = {
            "system_id": "B",
            "seed": 123,
            "model_module": "transkun.ModelTransformerLegacy",
            "model_config": {"baseSize": 64},
            "training": {
                "optimizer_steps": 10,
                "strict_pretrained": True,
            },
            "dataset_root": "dataset",
            "files": {
                "model_config": {"path": "b.conf", "sha256": "1"},
                "train_metadata": {"path": "train", "sha256": "2"},
                "validation_metadata": {"path": "val", "sha256": "3"},
                "pretrained_checkpoint": {"path": "pretrained", "sha256": "4"},
            },
            "git": {"commit": "abc", "dirty": False},
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "checkpoint.pt"
            self.create_checkpoint(path, metadata)

            proposed = dict(metadata)
            proposed["files"] = dict(metadata["files"])
            proposed["files"].pop("pretrained_checkpoint")
            proposed["training"] = dict(metadata["training"])
            proposed["training"]["strict_pretrained"] = False
            resumed = resolve_run_metadata(path, proposed)
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)

        self.assertEqual(resumed, metadata)
        self.assertEqual(checkpoint["metadata"], metadata)

    def test_resume_rejects_a_changed_seed(self):
        metadata = {
            "system_id": "C", "seed": 123, "model_module": "module",
            "model_config": {}, "training": {}, "dataset_root": "dataset",
            "files": {
                "model_config": {}, "train_metadata": {},
                "validation_metadata": {},
            },
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "checkpoint.pt"
            self.create_checkpoint(path, metadata)
            proposed = dict(metadata)
            proposed["seed"] = 124

            with self.assertRaisesRegex(ValueError, "seed"):
                resolve_run_metadata(path, proposed)


if __name__ == "__main__":
    unittest.main()
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from transkun.train import (
    build_training_state,
    capture_rng_state,
    checkpoint_resume_epoch,
    effective_data_loader_workers,
    periodic_checkpoint_due,
    remaining_epoch_indices,
    restore_rng_state,
    restore_training_state,
    scheduler_warmup_complete,
    seed_everything,
)
from transkun.TrainUtil import MovingBuffer, save_checkpoint


class TrainingReproducibilityTest(unittest.TestCase):
    def test_periodic_checkpoint_uses_completed_global_steps(self):
        self.assertFalse(periodic_checkpoint_due(0))
        self.assertFalse(periodic_checkpoint_due(499))
        self.assertTrue(periodic_checkpoint_due(500))
        self.assertTrue(periodic_checkpoint_due(2000))
        self.assertFalse(periodic_checkpoint_due(2001))

    def test_remaining_epoch_indices_resume_at_exact_batch_cursor(self):
        self.assertEqual(
            list(remaining_epoch_indices(11, batch_size=2, next_batch=3)),
            [6, 7, 8, 9])
        with self.assertRaisesRegex(ValueError, "cursor"):
            remaining_epoch_indices(10, batch_size=2, next_batch=6)

    def test_windows_uses_synchronous_data_loading(self):
        self.assertEqual(effective_data_loader_workers(2, platform="win32"), 0)
        self.assertEqual(effective_data_loader_workers(0, platform="win32"), 0)

    def test_non_windows_preserves_requested_workers(self):
        self.assertEqual(effective_data_loader_workers(2, platform="linux"), 2)

    def test_scheduler_warmup_is_absolute_across_resumes(self):
        self.assertFalse(scheduler_warmup_complete(500))
        self.assertTrue(scheduler_warmup_complete(501))
        self.assertTrue(scheduler_warmup_complete(6000))

    def test_periodic_checkpoint_resumes_in_the_current_epoch(self):
        self.assertEqual(
            checkpoint_resume_epoch(current_epoch=0, epoch_complete=False), 0)
        self.assertEqual(
            checkpoint_resume_epoch(current_epoch=4, epoch_complete=False), 4)

    def test_completed_epoch_checkpoint_advances_to_the_next_epoch(self):
        self.assertEqual(
            checkpoint_resume_epoch(current_epoch=0, epoch_complete=True), 1)
        self.assertEqual(
            checkpoint_resume_epoch(current_epoch=4, epoch_complete=True), 5)

    def test_declared_seed_reproduces_python_numpy_and_torch_rngs(self):
        seed_everything(1234, worker_id=2)
        first = (
            random.random(),
            np.random.random(),
            torch.rand(3),
        )

        seed_everything(1234, worker_id=2)
        second = (
            random.random(),
            np.random.random(),
            torch.rand(3),
        )

        self.assertEqual(first[0], second[0])
        self.assertEqual(first[1], second[1])
        torch.testing.assert_close(first[2], second[2])

    def test_worker_id_produces_a_distinct_stream(self):
        seed_everything(1234, worker_id=0)
        first = torch.rand(3)
        seed_everything(1234, worker_id=1)
        second = torch.rand(3)

        self.assertFalse(torch.equal(first, second))

    def test_rng_state_round_trip(self):
        seed_everything(4321)
        state = capture_rng_state()
        expected = (random.random(), np.random.random(), torch.rand(3))
        restore_rng_state(state)
        actual = (random.random(), np.random.random(), torch.rand(3))
        self.assertEqual(expected[0], actual[0])
        self.assertEqual(expected[1], actual[1])
        torch.testing.assert_close(expected[2], actual[2])

    def test_training_state_restores_cursor_history_losses_and_rng(self):
        seed_everything(99)
        history = MovingBuffer(initValue=40, maxLen=10000)
        history.step(12.5)
        state = build_training_state(3, 250, history, [1.5, 2.5])
        expected_random = random.random()
        next_batch, restored_history, losses = restore_training_state(state, 3)
        self.assertEqual(next_batch, 250)
        self.assertEqual(list(restored_history.values), [40, 12.5])
        self.assertEqual(losses, [1.5, 2.5])
        self.assertEqual(random.random(), expected_random)

    def test_training_state_rejects_legacy_checkpoint(self):
        with self.assertRaisesRegex(ValueError, "predates exact-resume"):
            restore_training_state({}, 0)

    def test_interrupted_resume_matches_uninterrupted_updates(self):
        inputs = [torch.tensor([[float(i), float(i + 1)]]) for i in range(6)]
        targets = [torch.tensor([[float(i) / 3]]) for i in range(6)]

        def create_components():
            model = torch.nn.Sequential(
                torch.nn.Linear(2, 4), torch.nn.Dropout(0.25),
                torch.nn.Linear(4, 1))
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=1e-3, total_steps=6)
            return model, optimizer, scheduler

        def update(model, optimizer, scheduler, index, history, losses):
            optimizer.zero_grad()
            prediction = model(inputs[index])
            loss = torch.nn.functional.mse_loss(prediction, targets[index])
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), history.getQuantile(0.8))
            history.step(norm.item())
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())

        seed_everything(2468)
        expected_model, expected_optimizer, expected_scheduler = create_components()
        expected_history = MovingBuffer(initValue=40, maxLen=10000)
        expected_losses = []
        for index in range(6):
            update(expected_model, expected_optimizer, expected_scheduler,
                   index, expected_history, expected_losses)

        seed_everything(2468)
        model, optimizer, scheduler = create_components()
        history = MovingBuffer(initValue=40, maxLen=10000)
        losses = []
        for index in range(3):
            update(model, optimizer, scheduler, index, history, losses)
        training_state = build_training_state(0, 3, history, losses)

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "resume.pt"
            save_checkpoint(
                checkpoint_path, 0, 3, model, {"train": [], "val": []},
                model.state_dict(), optimizer, scheduler,
                training_state=training_state)
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False)

        seed_everything(9999)
        resumed_model, resumed_optimizer, resumed_scheduler = create_components()
        resumed_model.load_state_dict(checkpoint["state_dict"])
        resumed_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        resumed_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        next_batch, resumed_history, resumed_losses = restore_training_state(
            checkpoint["training_state"], checkpoint["epoch"])
        for index in range(next_batch, 6):
            update(resumed_model, resumed_optimizer, resumed_scheduler,
                   index, resumed_history, resumed_losses)

        for expected, actual in zip(
                expected_model.parameters(), resumed_model.parameters()):
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        self.assertEqual(expected_losses, resumed_losses)
        self.assertEqual(
            expected_scheduler.state_dict(), resumed_scheduler.state_dict())
        self.assertEqual(
            list(expected_history.values), list(resumed_history.values))


if __name__ == "__main__":
    unittest.main()
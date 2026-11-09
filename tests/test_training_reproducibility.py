import random
import unittest

import numpy as np
import torch

from transkun.train import (
    checkpoint_resume_epoch,
    effective_data_loader_workers,
    scheduler_warmup_complete,
    seed_everything,
)


class TrainingReproducibilityTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
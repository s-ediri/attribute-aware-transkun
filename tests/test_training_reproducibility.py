import random
import unittest

import numpy as np
import torch

from transkun.train import seed_everything


class TrainingReproducibilityTest(unittest.TestCase):
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
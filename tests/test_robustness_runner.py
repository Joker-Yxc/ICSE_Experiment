import unittest

import numpy as np

from experiments.runners.run_escapture_robustness import perturb_sequence, stable_rng


class RobustnessPerturbationTests(unittest.TestCase):
    def setUp(self):
        self.sequence = [f"api_{index}" for index in range(20)]
        self.benign = ["benign_a", "benign_b"]
        self.probabilities = np.asarray([0.75, 0.25])

    def apply(self, perturbation, intensity=0.2):
        return perturb_sequence(
            self.sequence,
            perturbation,
            intensity,
            stable_rng(7, "sample", perturbation, intensity),
            self.benign,
            self.probabilities,
        )

    def test_insertion_adds_requested_fraction(self):
        self.assertEqual(len(self.apply("insertion")), 24)

    def test_deletion_removes_requested_fraction(self):
        self.assertEqual(len(self.apply("deletion")), 16)

    def test_local_reordering_preserves_multiset(self):
        reordered = self.apply("local_reordering")
        self.assertEqual(len(reordered), len(self.sequence))
        self.assertCountEqual(reordered, self.sequence)
        self.assertNotEqual(reordered, self.sequence)

    def test_perturbation_is_deterministic(self):
        self.assertEqual(self.apply("insertion"), self.apply("insertion"))


if __name__ == "__main__":
    unittest.main()

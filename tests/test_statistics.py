import unittest

import numpy as np

from cropstate.statistics import holm_adjust, paired_bootstrap_ci, paired_wilcoxon


class PairedBootstrapCiTests(unittest.TestCase):
    def test_identical_arrays_give_zero_mean_difference(self):
        a = np.array([1.0, 2.0, 3.0, 4.0])
        mean_diff, ci = paired_bootstrap_ci(a, a, iterations=200)
        self.assertAlmostEqual(mean_diff, 0.0)
        self.assertAlmostEqual(ci[0], 0.0)
        self.assertAlmostEqual(ci[1], 0.0)

    def test_constant_offset_is_recovered(self):
        a = np.array([5.0, 6.0, 7.0, 8.0, 9.0])
        b = a - 2.0
        mean_diff, ci = paired_bootstrap_ci(a, b, iterations=500, seed=1)
        self.assertAlmostEqual(mean_diff, 2.0)
        self.assertLessEqual(ci[0], mean_diff)
        self.assertGreaterEqual(ci[1], mean_diff)

    def test_raises_on_shape_mismatch(self):
        with self.assertRaises(ValueError):
            paired_bootstrap_ci(np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]))

    def test_deterministic_given_seed(self):
        a = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
        b = np.array([2.0, 2.0, 3.0, 4.0, 4.0])
        first = paired_bootstrap_ci(a, b, iterations=300, seed=7)
        second = paired_bootstrap_ci(a, b, iterations=300, seed=7)
        self.assertEqual(first, second)


class PairedWilcoxonTests(unittest.TestCase):
    def test_returns_statistic_and_p_value(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([2.0, 1.0, 4.0, 3.0, 6.0])
        result = paired_wilcoxon(a, b)
        self.assertIn("statistic", result)
        self.assertIn("p_value", result)
        self.assertGreaterEqual(result["p_value"], 0.0)
        self.assertLessEqual(result["p_value"], 1.0)


class HolmAdjustTests(unittest.TestCase):
    def test_adjusted_values_are_non_decreasing_with_rank(self):
        p_values = [0.01, 0.02, 0.03]
        adjusted = holm_adjust(p_values)
        self.assertEqual(len(adjusted), len(p_values))
        for value in adjusted:
            self.assertLessEqual(value, 1.0)

    def test_single_p_value_unchanged(self):
        adjusted = holm_adjust([0.04])
        self.assertAlmostEqual(adjusted[0], 0.04)

    def test_adjusted_values_are_at_least_as_large_as_raw(self):
        p_values = [0.2, 0.01, 0.05]
        adjusted = holm_adjust(p_values)
        for raw, corrected in zip(p_values, adjusted):
            self.assertGreaterEqual(corrected, raw - 1e-12)


if __name__ == "__main__":
    unittest.main()

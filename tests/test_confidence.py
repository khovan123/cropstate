import math
import unittest

import numpy as np

from cropstate.confidence import (
    agreement_from_jsd,
    combined_confidence,
    entropy_concentration,
    normalize_probability,
    top_two_margin,
)


class NormalizeProbabilityTests(unittest.TestCase):
    def test_sums_to_one(self):
        result = normalize_probability(np.array([1.0, 2.0, 3.0, 4.0]))
        self.assertAlmostEqual(result.sum(), 1.0)

    def test_clips_nonpositive_values(self):
        result = normalize_probability(np.array([0.0, 1.0, -1.0]))
        self.assertTrue(np.all(result > 0.0))


class EntropyConcentrationTests(unittest.TestCase):
    def test_uniform_distribution_has_zero_concentration(self):
        uniform = np.ones(6) / 6
        self.assertAlmostEqual(entropy_concentration(uniform), 0.0, places=6)

    def test_one_hot_distribution_has_full_concentration(self):
        one_hot = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.assertAlmostEqual(entropy_concentration(one_hot), 1.0, places=3)

    def test_bounded_between_zero_and_one(self):
        belief = np.array([0.5, 0.3, 0.1, 0.05, 0.03, 0.02])
        value = entropy_concentration(belief)
        self.assertGreaterEqual(value, 0.0)
        self.assertLessEqual(value, 1.0)


class TopTwoMarginTests(unittest.TestCase):
    def test_one_hot_has_full_margin(self):
        belief = np.array([1.0, 0.0, 0.0])
        self.assertAlmostEqual(top_two_margin(belief), 1.0)

    def test_two_way_tie_has_zero_margin(self):
        belief = np.array([0.5, 0.5, 0.0])
        self.assertAlmostEqual(top_two_margin(belief), 0.0)

    def test_order_independent(self):
        first = top_two_margin(np.array([0.1, 0.7, 0.2]))
        second = top_two_margin(np.array([0.7, 0.1, 0.2]))
        self.assertAlmostEqual(first, second)


class AgreementFromJsdTests(unittest.TestCase):
    def test_identical_distributions_have_full_agreement(self):
        belief = np.array([0.6, 0.4])
        self.assertAlmostEqual(agreement_from_jsd(belief, belief), 1.0, places=6)

    def test_disjoint_distributions_have_zero_agreement(self):
        p = np.array([1.0, 0.0])
        q = np.array([0.0, 1.0])
        self.assertAlmostEqual(agreement_from_jsd(p, q), 0.0, places=6)


class CombinedConfidenceTests(unittest.TestCase):
    def test_ignores_agreement_weight_without_temporal(self):
        belief = np.array([0.7, 0.1, 0.1, 0.05, 0.03, 0.02])
        value = combined_confidence(belief, temporal=None, eta_entropy=0.6, eta_margin=0.4, eta_agreement=0.5)
        expected = 0.6 * entropy_concentration(belief) + 0.4 * top_two_margin(belief)
        self.assertAlmostEqual(value, expected, places=6)

    def test_uses_agreement_when_temporal_present(self):
        belief = np.array([0.7, 0.1, 0.1, 0.05, 0.03, 0.02])
        temporal = np.array([0.1, 0.7, 0.1, 0.05, 0.03, 0.02])
        with_temporal = combined_confidence(belief, temporal=temporal, eta_agreement=0.5)
        without_temporal = combined_confidence(belief, temporal=None, eta_agreement=0.5)
        self.assertNotAlmostEqual(with_temporal, without_temporal, places=6)

    def test_raises_when_all_weights_nonpositive(self):
        belief = np.array([0.5, 0.5])
        with self.assertRaises(ValueError):
            combined_confidence(belief, eta_entropy=0.0, eta_margin=0.0, eta_agreement=0.0)

    def test_result_is_bounded(self):
        belief = np.array([0.5, 0.5])
        value = combined_confidence(belief)
        self.assertTrue(0.0 <= value <= 1.0)
        self.assertFalse(math.isnan(value))


if __name__ == "__main__":
    unittest.main()

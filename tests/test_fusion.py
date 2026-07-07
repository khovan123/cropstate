import unittest

import numpy as np

from cropstate.fusion import log_linear_fusion, softmax, transition_prior


class SoftmaxTests(unittest.TestCase):
    def test_sums_to_one(self):
        result = softmax(np.array([1.0, 2.0, 3.0]))
        self.assertAlmostEqual(result.sum(), 1.0)

    def test_is_shift_invariant(self):
        first = softmax(np.array([1.0, 2.0, 3.0]))
        second = softmax(np.array([101.0, 102.0, 103.0]))
        np.testing.assert_allclose(first, second, atol=1e-8)

    def test_uniform_logits_give_uniform_output(self):
        result = softmax(np.zeros(4))
        np.testing.assert_allclose(result, np.ones(4) / 4)


class TransitionPriorTests(unittest.TestCase):
    def test_identity_matrix_preserves_belief(self):
        belief = np.array([0.2, 0.3, 0.5])
        identity = np.eye(3)
        result = transition_prior(belief, identity)
        np.testing.assert_allclose(result, belief)

    def test_result_sums_to_one(self):
        belief = np.array([0.2, 0.3, 0.5])
        transition = np.array([
            [0.7, 0.3, 0.0],
            [0.0, 0.6, 0.4],
            [0.0, 0.0, 1.0],
        ])
        result = transition_prior(belief, transition)
        self.assertAlmostEqual(result.sum(), 1.0)


class LogLinearFusionTests(unittest.TestCase):
    def test_output_is_valid_probability_distribution(self):
        prior = np.array([0.5, 0.3, 0.2])
        evidence = [np.array([0.1, 0.1, 0.8])]
        result = log_linear_fusion(prior, evidence, [1.0])
        self.assertAlmostEqual(result.sum(), 1.0)
        self.assertTrue(np.all(result >= 0.0))

    def test_agreeing_evidence_reinforces_prior_mode(self):
        prior = np.array([0.5, 0.3, 0.2])
        evidence = [np.array([0.9, 0.05, 0.05])]
        result = log_linear_fusion(prior, evidence, [2.0])
        self.assertGreater(result[0], prior[0])

    def test_raises_on_mismatched_evidence_and_weight_counts(self):
        prior = np.array([0.5, 0.5])
        with self.assertRaises(ValueError):
            log_linear_fusion(prior, [np.array([0.5, 0.5]), np.array([0.4, 0.6])], [1.0])


if __name__ == "__main__":
    unittest.main()

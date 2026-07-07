import unittest

import numpy as np

from cropstate.metrics import (
    dcg,
    expected_calibration_error,
    multiclass_brier,
    ndcg_at_k,
    precision_recall_at_k,
    sirr_at_k,
    vision_metrics,
)


class VisionMetricsTests(unittest.TestCase):
    def test_perfect_predictions(self):
        labels = np.array([0, 1, 2, 1, 0])
        result = vision_metrics(labels, labels)
        self.assertAlmostEqual(result["accuracy"], 1.0)
        self.assertAlmostEqual(result["macro_f1"], 1.0)
        self.assertAlmostEqual(result["masd"], 0.0)

    def test_masd_reflects_stage_distance(self):
        y_true = np.array([0, 5])
        y_pred = np.array([1, 0])
        result = vision_metrics(y_true, y_pred)
        self.assertAlmostEqual(result["masd"], 3.0)


class CalibrationTests(unittest.TestCase):
    def test_perfectly_calibrated_predictions_have_zero_ece(self):
        # 9/10 confidence-0.9 predictions correct: accuracy (0.9) matches confidence (0.9).
        probabilities = np.array([[0.9, 0.1]] * 9 + [[0.1, 0.9]])
        labels = np.array([0] * 9 + [0])
        ece = expected_calibration_error(probabilities, labels, bins=10)
        self.assertLess(ece, 1e-6)

    def test_overconfident_wrong_predictions_increase_ece(self):
        probabilities = np.array([[0.95, 0.05], [0.95, 0.05]])
        labels = np.array([1, 1])
        ece = expected_calibration_error(probabilities, labels, bins=10)
        self.assertGreater(ece, 0.5)


class BrierTests(unittest.TestCase):
    def test_confident_correct_prediction_has_zero_brier(self):
        probabilities = np.array([[1.0, 0.0, 0.0]])
        labels = np.array([0])
        self.assertAlmostEqual(multiclass_brier(probabilities, labels), 0.0)

    def test_confident_wrong_prediction_has_high_brier(self):
        probabilities = np.array([[0.0, 0.0, 1.0]])
        labels = np.array([0])
        self.assertAlmostEqual(multiclass_brier(probabilities, labels), 2.0)


class RankingMetricsTests(unittest.TestCase):
    def test_dcg_empty_relevances(self):
        self.assertEqual(dcg([], 5), 0.0)

    def test_ndcg_perfect_ranking_is_one(self):
        ranked = ["a", "b", "c"]
        relevance = {"a": 3.0, "b": 2.0, "c": 1.0}
        self.assertAlmostEqual(ndcg_at_k(ranked, relevance, k=3), 1.0)

    def test_ndcg_with_no_relevant_documents_is_zero(self):
        ranked = ["a", "b"]
        self.assertEqual(ndcg_at_k(ranked, {}, k=2), 0.0)

    def test_precision_recall_at_k(self):
        ranked = ["a", "b", "c", "d"]
        relevant = {"a", "c"}
        precision, recall = precision_recall_at_k(ranked, relevant, k=2)
        self.assertAlmostEqual(precision, 0.5)
        self.assertAlmostEqual(recall, 0.5)

    def test_precision_recall_with_no_relevant_ids(self):
        precision, recall = precision_recall_at_k(["a", "b"], set(), k=2)
        self.assertAlmostEqual(precision, 0.0)
        self.assertAlmostEqual(recall, 0.0)

    def test_sirr_counts_incompatible_documents_in_topk(self):
        ranked = ["a", "b", "c", "d"]
        incompatible = {"b", "d"}
        self.assertAlmostEqual(sirr_at_k(ranked, incompatible, k=4), 0.5)

    def test_sirr_is_zero_when_no_incompatible_documents(self):
        self.assertAlmostEqual(sirr_at_k(["a", "b"], set(), k=2), 0.0)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def vision_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "masd": float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred)))),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def expected_calibration_error(probabilities: np.ndarray, labels: np.ndarray, bins: int = 15) -> float:
    probabilities = np.asarray(probabilities)
    labels = np.asarray(labels)
    confidences = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.any():
            accuracy = np.mean(predictions[mask] == labels[mask])
            ece += mask.mean() * abs(accuracy - confidences[mask].mean())
    return float(ece)


def multiclass_brier(probabilities: np.ndarray, labels: np.ndarray) -> float:
    one_hot = np.eye(probabilities.shape[1])[labels]
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def dcg(relevances: list[float], k: int) -> float:
    rel = np.asarray(relevances[:k], dtype=float)
    if rel.size == 0:
        return 0.0
    return float(np.sum((2**rel - 1) / np.log2(np.arange(2, rel.size + 2))))


def ndcg_at_k(ranked_ids: list[str], relevance: dict[str, float], k: int = 5) -> float:
    actual = [relevance.get(doc_id, 0.0) for doc_id in ranked_ids[:k]]
    ideal = sorted(relevance.values(), reverse=True)[:k]
    denominator = dcg(ideal, k)
    return 0.0 if denominator == 0 else dcg(actual, k) / denominator


def precision_recall_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int = 5) -> tuple[float, float]:
    top = ranked_ids[:k]
    hits = sum(doc_id in relevant_ids for doc_id in top)
    return hits / k, (hits / len(relevant_ids) if relevant_ids else 0.0)


def sirr_at_k(ranked_ids: list[str], incompatibles: set[str], k: int = 5) -> float:
    top = ranked_ids[:k]
    return sum(doc_id in incompatibles for doc_id in top) / k

from __future__ import annotations

import numpy as np


def softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    e = np.exp(x - x.max())
    return e / e.sum()


def transition_prior(previous_belief: np.ndarray, transition_matrix: np.ndarray) -> np.ndarray:
    prior = np.asarray(transition_matrix, dtype=float).T @ np.asarray(previous_belief, dtype=float)
    return prior / prior.sum()


def log_linear_fusion(
    prior: np.ndarray,
    evidence_vectors: list[np.ndarray],
    evidence_weights: list[float],
    prior_weight: float = 1.0,
    eps: float = 1e-9,
) -> np.ndarray:
    if len(evidence_vectors) != len(evidence_weights):
        raise ValueError("Evidence and weight counts differ")
    z = prior_weight * np.log(np.clip(prior, eps, 1.0))
    for p, weight in zip(evidence_vectors, evidence_weights, strict=True):
        p = np.asarray(p, dtype=float)
        p = p / p.sum()
        z += weight * np.log(np.clip(p, eps, 1.0))
    return softmax(z)

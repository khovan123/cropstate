from __future__ import annotations

import numpy as np
from scipy.spatial.distance import jensenshannon


def normalize_probability(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    p = np.clip(p, eps, None)
    return p / p.sum()


def entropy_concentration(p: np.ndarray) -> float:
    p = normalize_probability(p)
    entropy = -float(np.sum(p * np.log(p)))
    return float(np.clip(1.0 - entropy / np.log(len(p)), 0.0, 1.0))


def top_two_margin(p: np.ndarray) -> float:
    p = np.sort(normalize_probability(p))[::-1]
    return float(p[0] - p[1])


def agreement_from_jsd(p: np.ndarray, q: np.ndarray) -> float:
    p, q = normalize_probability(p), normalize_probability(q)
    jsd = float(jensenshannon(p, q, base=np.e) ** 2)
    return float(np.clip(1.0 - jsd / np.log(2.0), 0.0, 1.0))


def combined_confidence(
    belief: np.ndarray,
    temporal: np.ndarray | None = None,
    eta_entropy: float = 0.6,
    eta_margin: float = 0.4,
    eta_agreement: float = 0.0,
) -> float:
    weights = np.array([eta_entropy, eta_margin, eta_agreement], dtype=float)
    if temporal is None:
        weights[2] = 0.0
    if weights.sum() <= 0:
        raise ValueError("At least one confidence weight must be positive")
    weights /= weights.sum()
    values = [entropy_concentration(belief), top_two_margin(belief), 1.0]
    if temporal is not None:
        values[2] = agreement_from_jsd(belief, temporal)
    return float(np.dot(weights, values))

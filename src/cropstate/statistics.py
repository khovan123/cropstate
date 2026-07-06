from __future__ import annotations

import numpy as np
from scipy.stats import wilcoxon


def paired_bootstrap_ci(a, b, iterations: int = 10000, seed: int = 42, confidence: float = 0.95):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("Paired arrays must have equal shape")
    rng = np.random.default_rng(seed)
    differences = a - b
    samples = np.empty(iterations)
    for i in range(iterations):
        idx = rng.integers(0, len(differences), len(differences))
        samples[i] = differences[idx].mean()
    alpha = 1 - confidence
    return float(differences.mean()), tuple(np.quantile(samples, [alpha / 2, 1 - alpha / 2]))


def paired_wilcoxon(a, b):
    result = wilcoxon(np.asarray(a), np.asarray(b), zero_method="wilcox", alternative="two-sided")
    return {"statistic": float(result.statistic), "p_value": float(result.pvalue)}


def holm_adjust(p_values: list[float]) -> list[float]:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, idx in enumerate(order):
        value = min(1.0, (m - rank) * p[idx])
        running = max(running, value)
        adjusted[idx] = running
    return adjusted.tolist()

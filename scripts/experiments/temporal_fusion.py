"""Phenology-constrained temporal stage-belief update (Tier B#5).

Rice phenology is monotonic: a field does not return from ripening to tillering.
This exercises the existing fusion.py machinery (transition_prior + log_linear_fusion)
to show that a phenology transition prior corrects the network's worst mistakes
(non-adjacent stage jumps) that a single-frame argmax makes.

Honesty note: the pilot has no real image time-series, so this is a CONTROLLED
demonstration. Real per-image beliefs are arranged into monotonic trajectories
(ordered by true stage, with within-stage shuffling) and the prior is applied
online. It demonstrates the mechanism and its effect, not field-deployed tracking.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cropstate.constants import STAGE_NAMES
from cropstate.fusion import log_linear_fusion, transition_prior


def softmax_np(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def phenology_transition_matrix(stay=0.55, adv1=0.35, adv2=0.10) -> np.ndarray:
    n = len(STAGE_NAMES)
    T = np.zeros((n, n))
    for i in range(n):
        T[i, i] = stay
        if i + 1 < n:
            T[i, i + 1] = adv1
        if i + 2 < n:
            T[i, i + 2] = adv2
        T[i] = T[i] / T[i].sum()
    return T


def errors(preds, labels):
    diff = np.abs(preds - labels)
    return {
        "accuracy": float(np.mean(preds == labels)),
        "masd": float(np.mean(diff)),
        "non_adjacent_errors": int(np.sum(diff >= 2)),
        "adjacent_errors": int(np.sum(diff == 1)),
    }


def run_trajectory(beliefs, labels, T, prior_weight, evidence_weight, seed=42):
    """Order images into a monotonic phenology trajectory and fuse online."""
    rng = np.random.default_rng(seed)
    order = np.lexsort((rng.random(len(labels)), labels))  # by true stage, shuffle within
    fused_preds = np.empty(len(labels), dtype=int)
    prev_belief = None
    for pos in order:
        evidence = beliefs[pos]
        if prev_belief is None:
            fused = evidence
        else:
            prior = transition_prior(prev_belief, T)
            fused = log_linear_fusion(prior, [evidence], [evidence_weight], prior_weight=prior_weight)
        fused_preds[pos] = int(fused.argmax())
        prev_belief = fused
    return fused_preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logits-index", default="CROPSTATE_RESULTS/novelty/logits/index.json")
    parser.add_argument("--model", default="resnet18")
    parser.add_argument("--prior-weight", type=float, default=1.0)
    parser.add_argument("--evidence-weight", type=float, default=1.0)
    parser.add_argument("--output", default="CROPSTATE_RESULTS/novelty/temporal_fusion.json")
    args = parser.parse_args()

    index = json.loads(Path(args.logits_index).read_text())
    data = np.load(index[args.model]["path"], allow_pickle=True)
    beliefs = softmax_np(data["test_logits"].astype(np.float64))
    labels = data["test_labels"].astype(int)

    T = phenology_transition_matrix()
    raw_preds = beliefs.argmax(1)
    raw = errors(raw_preds, labels)

    # Average over several random within-stage orderings for stability.
    fused_metrics = []
    for seed in range(20):
        fp = run_trajectory(beliefs, labels, T, args.prior_weight, args.evidence_weight, seed=seed)
        fused_metrics.append(errors(fp, labels))
    fused_mean = {k: float(np.mean([m[k] for m in fused_metrics])) for k in raw}

    payload = {
        "model": args.model,
        "note": "controlled demonstration on monotonic trajectories (no real time-series)",
        "transition_matrix": T.tolist(),
        "single_frame_argmax": raw,
        "temporal_fused_mean_over_20_orderings": fused_mean,
        "non_adjacent_error_reduction": raw["non_adjacent_errors"] - fused_mean["non_adjacent_errors"],
        "masd_reduction": raw["masd"] - fused_mean["masd"],
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps({k: payload[k] for k in
                      ["single_frame_argmax", "temporal_fused_mean_over_20_orderings",
                       "non_adjacent_error_reduction", "masd_reduction"]}, indent=2))


if __name__ == "__main__":
    main()

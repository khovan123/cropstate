"""McNemar's exact test on the shared test split (Tier C#7).

The paper states a confirmatory replication "should additionally report McNemar's
test". This computes it from the stored per-image correctness vectors in
vision_baseline_comparison.json, with Holm correction across model pairs.
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from math import comb
from pathlib import Path

from cropstate.statistics import holm_adjust


def mcnemar_exact(correct_a: list[int], correct_b: list[int]) -> dict:
    """Exact binomial McNemar test on discordant pairs (two-sided)."""
    b = sum(1 for x, y in zip(correct_a, correct_b) if x == 1 and y == 0)  # a right, b wrong
    c = sum(1 for x, y in zip(correct_a, correct_b) if x == 0 and y == 1)  # a wrong, b right
    n = b + c
    if n == 0:
        p = 1.0
    else:
        k = min(b, c)
        tail = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
        p = min(1.0, 2.0 * tail)
    return {"b_only_a_correct": b, "c_only_b_correct": c, "discordant": n, "p_value": p}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", default="CROPSTATE_RESULTS/vision_baseline_comparison.json")
    parser.add_argument("--output", default="CROPSTATE_RESULTS/novelty/mcnemar.json")
    args = parser.parse_args()

    data = json.loads(Path(args.comparison).read_text())
    correctness = data["per_image_correctness"]
    models = list(correctness)

    pairs = list(combinations(models, 2))
    results = []
    for a, b in pairs:
        res = mcnemar_exact(correctness[a], correctness[b])
        res.update({"model_a": a, "model_b": b,
                    "accuracy_a": sum(correctness[a]) / len(correctness[a]),
                    "accuracy_b": sum(correctness[b]) / len(correctness[b])})
        results.append(res)
    adjusted = holm_adjust([r["p_value"] for r in results])
    for r, p_adj in zip(results, adjusted):
        r["holm_adjusted_p_value"] = p_adj
        r["significant_holm_0.05"] = p_adj < 0.05

    payload = {"test": "mcnemar_exact_binomial", "n_test_images": len(next(iter(correctness.values()))),
               "pairs": results}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    for r in results:
        print(f"{r['model_a']:>16} vs {r['model_b']:<16} "
              f"b={r['b_only_a_correct']:>2} c={r['c_only_b_correct']:>2} "
              f"p={r['p_value']:.4f} p_holm={r['holm_adjusted_p_value']:.4f} "
              f"{'*' if r['significant_holm_0.05'] else ''}")


if __name__ == "__main__":
    main()

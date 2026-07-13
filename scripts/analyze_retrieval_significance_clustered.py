"""Image-clustered paired-bootstrap significance and error-stratified retrieval analysis.

Addresses two reviewer-mandated (P0) fixes for the CropState-Vision paper:

  * P0-3.8  Retrieval "significantly outperforms" claims need paired difference
            tables with 95% CIs and Holm-adjusted p-values. Because a single test
            image spawns several topic scenarios, the bootstrap resamples whole
            *images* (clusters of scenarios) rather than individual scenarios, so
            the CIs respect the non-independence of same-image scenarios.
  * P0-3.3  The error-stratified table (correct / adjacent / non-adjacent) must
            include the strongest baseline B1 (query expansion) for a fair
            comparison against the proposed adaptive gating (P).

Input : CROPSTATE_RESULTS/retrieval/retrieval_evaluation_belief.json
Output: CROPSTATE_RESULTS/retrieval/retrieval_significance_clustered.json
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

RESULTS = Path("CROPSTATE_RESULTS/retrieval")
IN_PATH = RESULTS / "retrieval_evaluation_belief.json"
OUT_PATH = RESULTS / "retrieval_significance_clustered.json"

STAGE_ORDER = {
    "establishment": 0,
    "tillering": 1,
    "stem_booting": 2,
    "reproductive": 3,
    "grain_filling": 4,
    "ripening": 5,
}
METHODS = [
    "B0_ungated",
    "B1_query_expansion",
    "B2_hard_filter",
    "B3_fixed_soft",
    "P_adaptive_soft",
    "oracle_reference",
]
METRICS = ["ndcg_at_k", "recall_at_k", "precision_at_k", "sirr_at_k"]
PROPOSED = "P_adaptive_soft"
BASELINES = ["B0_ungated", "B1_query_expansion", "B3_fixed_soft"]
N_BOOT = 5000
SEED = 42


def error_condition(row: dict) -> str:
    delta = abs(STAGE_ORDER[row["ground_truth_stage"]] - STAGE_ORDER[row["predicted_stage"]])
    if delta == 0:
        return "correct"
    if delta == 1:
        return "adjacent"
    return "nonadjacent"


def image_id(scenario_id: str) -> str:
    # scenario_id == f"{image_id}__{topic}"
    return scenario_id.split("__")[0]


def main() -> None:
    data = json.loads(IN_PATH.read_text())
    rows = data["rows"]

    by_scenario: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        by_scenario[row["scenario_id"]][row["method"]] = row
    scenarios = list(by_scenario)

    # Cluster scenarios by source image so the bootstrap resamples whole images.
    clusters: dict[str, list[str]] = defaultdict(list)
    for scenario in scenarios:
        clusters[image_id(scenario)].append(scenario)
    cluster_keys = list(clusters)

    rng = np.random.default_rng(SEED)

    def paired_bootstrap(proposed: str, baseline: str, metric: str) -> dict:
        diffs = {
            s: by_scenario[s][proposed][metric] - by_scenario[s][baseline][metric]
            for s in scenarios
        }
        observed = float(np.mean([diffs[s] for s in scenarios]))
        boot = np.empty(N_BOOT)
        n_clusters = len(cluster_keys)
        for b in range(N_BOOT):
            picks = rng.integers(0, n_clusters, n_clusters)
            vals = [diffs[s] for idx in picks for s in clusters[cluster_keys[idx]]]
            boot[b] = np.mean(vals)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        p = 2.0 * min(float(np.mean(boot <= 0)), float(np.mean(boot >= 0)))
        return {"diff": observed, "ci95": [float(lo), float(hi)], "p_raw": min(p, 1.0)}

    comparisons = {}
    flat = []
    for baseline in BASELINES:
        comparisons[baseline] = {}
        for metric in METRICS:
            res = paired_bootstrap(PROPOSED, baseline, metric)
            comparisons[baseline][metric] = res
            flat.append((baseline, metric, res))

    # Holm-Bonferroni across the whole family of tests.
    flat.sort(key=lambda t: t[2]["p_raw"])
    m = len(flat)
    running = 0.0
    for i, (baseline, metric, res) in enumerate(flat):
        adj = min(1.0, (m - i) * res["p_raw"])
        adj = max(adj, running)
        running = adj
        comparisons[baseline][metric]["p_holm"] = adj

    # Error-stratified nDCG/SIRR for every method (incl. B1).
    strata: dict[str, list[str]] = defaultdict(list)
    for scenario in scenarios:
        strata[error_condition(by_scenario[scenario]["B0_ungated"])].append(scenario)

    stratified = {}
    for condition, scs in strata.items():
        stratified[condition] = {"n": len(scs), "methods": {}}
        for method in METHODS:
            stratified[condition]["methods"][method] = {
                "ndcg_at_k": float(np.mean([by_scenario[s][method]["ndcg_at_k"] for s in scs])),
                "sirr_at_k": float(np.mean([by_scenario[s][method]["sirr_at_k"] for s in scs])),
            }

    out = {
        "n_scenarios": len(scenarios),
        "n_images": len(cluster_keys),
        "n_bootstrap": N_BOOT,
        "seed": SEED,
        "proposed": PROPOSED,
        "clustering": "scenarios resampled by source image id",
        "multiple_comparison_correction": "holm",
        "paired_bootstrap": comparisons,
        "error_stratified": stratified,
    }
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()

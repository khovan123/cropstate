"""Consolidate all novelty experiment outputs into one summary table."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("CROPSTATE_RESULTS/novelty")


def load(name):
    p = ROOT / name
    return json.loads(p.read_text()) if p.exists() else None


def main():
    lines = ["# CropState-Vision — novelty experiment results\n"]

    mc = load("mcnemar.json")
    if mc:
        lines.append("## Tier C#7 — McNemar exact test (shared 83-image split)")
        for r in mc["pairs"]:
            sig = "significant" if r["significant_holm_0.05"] else "n.s."
            lines.append(f"- {r['model_a']} vs {r['model_b']}: b={r['b_only_a_correct']} c={r['c_only_b_correct']} "
                         f"p={r['p_value']:.4f} p_holm={r['holm_adjusted_p_value']:.4f} ({sig})")
        lines.append("")

    lk = load("leakage_audit.json")
    if lk:
        lines.append("## Tier A#1 — Near-duplicate / leakage audit")
        lines.append(f"{lk['n_images']} images. Threshold sweep (Hamming over 64-bit avg-hash):")
        lines.append("| thr | dup edges | multi-img clusters | largest | straddling splits | cross-split pairs | test/val leak |")
        lines.append("|----|----|----|----|----|----|----|")
        for s in lk["threshold_sweep"]:
            lines.append(f"| {s['hamming_threshold']} | {s['near_duplicate_edges']} | {s['multi_image_clusters']} | "
                         f"{s['largest_cluster_size']} | {s['clusters_straddling_splits']} | "
                         f"{s['cross_split_pairs']} | {s['test_or_val_leak_pairs']} |")
        lines.append("")

    cal = load("calibration.json")
    if cal:
        lines.append("## Tier A#3 — Temperature scaling / calibration")
        lines.append("| model | T | ECE before | ECE after | Brier before | Brier after | NLL before | NLL after |")
        lines.append("|----|----|----|----|----|----|----|----|")
        for k, v in cal.items():
            b, a = v["test_before"], v["test_after"]
            lines.append(f"| {k} | {v['temperature']:.3f} | {b['ece']:.3f} | {a['ece']:.3f} | "
                         f"{b['brier']:.3f} | {a['brier']:.3f} | {b['nll']:.3f} | {a['nll']:.3f} |")
        lines.append("")

    tf = load("temporal_fusion.json")
    if tf:
        lines.append("## Tier B#5 — Phenology temporal stage-belief fusion (controlled)")
        raw, fused = tf["single_frame_argmax"], tf["temporal_fused_mean_over_20_orderings"]
        lines.append(f"- single-frame argmax: acc={raw['accuracy']:.3f} MASD={raw['masd']:.3f} "
                     f"non-adjacent errors={raw['non_adjacent_errors']}")
        lines.append(f"- temporal-fused (mean/20 orderings): acc={fused['accuracy']:.3f} MASD={fused['masd']:.3f} "
                     f"non-adjacent errors={fused['non_adjacent_errors']:.2f}")
        lines.append(f"- non-adjacent error reduction: {tf['non_adjacent_error_reduction']:.2f}; "
                     f"MASD reduction: {tf['masd_reduction']:.3f}")
        lines.append("")

    rg = load("retrieval_gating.json")
    if rg:
        lines.append("## Tier B#4 — Closed-loop stage-gated retrieval (SIRR@k, lower=better)")
        lines.append("| method | precision@k | recall@k | nDCG@k | SIRR@k |")
        lines.append("|----|----|----|----|----|")
        for m, s in rg["summary"].items():
            lines.append(f"| {m} | {s['precision_at_k']:.3f} | {s['recall_at_k']:.3f} | "
                         f"{s['ndcg_at_k']:.3f} | {s['sirr_at_k']:.3f} |")
        lines.append("")

    rt = load("retrain_summary.json")
    if rt:
        lines.append("## Tier A#2 / C#6 / C#9 — Retrained variants (fixed split, vs CE baseline)")
        lines.append("| variant | accuracy | macro-F1 | MASD | non-adj err | reproductive recall | ECE |")
        lines.append("|----|----|----|----|----|----|----|")
        for v in rt:
            lines.append(f"| {v['variant']} | {v['accuracy']:.3f} | {v['macro_f1']:.3f} | {v['masd']:.3f} | "
                         f"{v.get('non_adjacent_errors','?')} | {v['reproductive_recall']:.3f} | {v['ece']:.3f} |")
        lines.append("")

    out = ROOT / "SUMMARY.md"
    out.write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()

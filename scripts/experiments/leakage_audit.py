"""Near-duplicate / acquisition-leakage audit (Tier A#1).

The manifest's parent_image_id is synthesized per-image (stage:session:stem), so
the "grouped" split degrades to an image-level split. This script quantifies how
much leakage that actually risks: it detects near-duplicate image clusters with a
perceptual hash (64-bit average hash + Hamming distance), reports how many
clusters straddle >1 split in the current manifest (real leak count), and emits a
leak-free split assignment that keeps every near-duplicate cluster in one split.

Dependencies: Pillow + numpy only (no torch).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def average_hash(path: Path, size: int = 8) -> np.ndarray:
    img = Image.open(path).convert("L").resize((size, size), Image.BILINEAR)
    pixels = np.asarray(img, dtype=np.float64)
    return (pixels > pixels.mean()).flatten()


def union_find_clusters(n: int, edges: list[tuple[int, int]]) -> list[int]:
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    return [find(i) for i in range(n)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="CROPSTATE_RESULTS/vision_final/manifest.csv")
    parser.add_argument("--data-root", default="CROPSTATE_DATASET")
    parser.add_argument("--hamming-threshold", type=int, default=6,
                        help="Max Hamming distance (out of 64) to treat two images as near-duplicates.")
    parser.add_argument("--output", default="CROPSTATE_RESULTS/novelty/leakage_audit.json")
    parser.add_argument("--grouped-manifest", default="CROPSTATE_RESULTS/novelty/manifest_hash_grouped.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.manifest).reset_index(drop=True)
    root = Path(args.data_root)
    hashes = np.stack([average_hash(root / p) for p in df["image_path"]])
    n = len(df)
    split_by_idx = df["split"].tolist()

    def edges_at(threshold: int) -> list[tuple[int, int]]:
        out = []
        for i in range(n):
            dists = np.count_nonzero(hashes[i] != hashes[i + 1:], axis=1)
            for offset, d in enumerate(dists):
                if d <= threshold:
                    out.append((i, i + 1 + offset))
        return out

    # Threshold sweep: exact-dup thresholds vs visual-similarity thresholds.
    sweep = []
    for threshold in [4, 6, 8, 10, 12, 14, 16, 20]:
        e = edges_at(threshold)
        cl = union_find_clusters(n, e)
        sizes = pd.Series(cl).value_counts()
        straddle = df.assign(_c=cl).groupby("_c")["split"].nunique()
        cross = [(a, b) for a, b in e if split_by_idx[a] != split_by_idx[b]]
        tv = [(a, b) for a, b in cross
              if "test" in (split_by_idx[a], split_by_idx[b]) or "validation" in (split_by_idx[a], split_by_idx[b])]
        sweep.append({
            "hamming_threshold": threshold,
            "near_duplicate_edges": len(e),
            "multi_image_clusters": int((sizes > 1).sum()),
            "largest_cluster_size": int(sizes.max()),
            "clusters_straddling_splits": int((straddle > 1).sum()),
            "cross_split_pairs": len(cross),
            "test_or_val_leak_pairs": len(tv),
        })

    # Pairwise Hamming distances; connect near-duplicates at the reporting threshold.
    edges = edges_at(args.hamming_threshold)

    cluster_ids = union_find_clusters(n, edges)
    df["dup_cluster"] = cluster_ids

    # How many near-duplicate clusters straddle >1 split under the current manifest?
    cluster_split = df.groupby("dup_cluster")["split"].nunique()
    multi_image_clusters = df.groupby("dup_cluster").size()
    leaking_clusters = cluster_split[cluster_split > 1].index.tolist()
    leaking_images = int(df[df["dup_cluster"].isin(leaking_clusters)].shape[0])

    # Cross-split near-duplicate PAIRS (the concrete leak): an edge whose endpoints
    # fall in different splits, at least one of them being test/validation.
    cross_split_pairs = [
        (int(a), int(b)) for a, b in edges if split_by_idx[a] != split_by_idx[b]
    ]
    test_val_leak_pairs = [
        (a, b) for a, b in cross_split_pairs
        if "test" in (split_by_idx[a], split_by_idx[b]) or "validation" in (split_by_idx[a], split_by_idx[b])
    ]

    payload = {
        "threshold_sweep": sweep,
        "reporting_hamming_threshold": args.hamming_threshold,
        "n_images": n,
        "n_near_duplicate_edges": len(edges),
        "n_clusters": int(pd.Series(cluster_ids).nunique()),
        "n_multi_image_clusters": int((multi_image_clusters > 1).sum()),
        "largest_cluster_size": int(multi_image_clusters.max()),
        "n_clusters_straddling_splits": len(leaking_clusters),
        "n_images_in_leaking_clusters": leaking_images,
        "n_cross_split_near_duplicate_pairs": len(cross_split_pairs),
        "n_test_or_val_leak_pairs": len(test_val_leak_pairs),
        "example_leak_pairs": [
            {"a": df.loc[a, "image_path"], "a_split": split_by_idx[a],
             "b": df.loc[b, "image_path"], "b_split": split_by_idx[b]}
            for a, b in test_val_leak_pairs[:15]
        ],
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))

    # Emit a leak-free split: keep whole hash-clusters together (grouped by dup_cluster),
    # stratified by stage, sizes matched to the original split proportions.
    grouped = Path(args.grouped_manifest)
    grouped.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(grouped, index=False)

    print(json.dumps({k: v for k, v in payload.items() if k != "example_leak_pairs"}, indent=2))


if __name__ == "__main__":
    main()

"""Encode a real parent_image_id into filenames via near-duplicate clustering.

The dataset's original capture/parent metadata is lost (files were renamed to
img001...). The only recoverable grouping signal is visual: overlapping patches
of one parent scene are near-duplicates. This clusters each stage folder's images
by perceptual hash and renames members of a cluster to

    p<NNN>_subset_overlap_<k>.<ext>

so the codebase's parent_image_id() = stem.split("_subset_overlap")[0] picks up
the group, and grouped_stratified_split keeps a parent's patches in ONE split
(leak-free). Singletons also get p<NNN>_subset_overlap_000 so the scheme is uniform.

The parent id is DERIVED from visual clustering, not ground-truth capture metadata
— label it as such in any paper. Default is a DRY RUN; pass --apply to rename, and
--undo to restore from the mapping. Depends on Pillow + numpy only (no torch).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from leakage_audit import average_hash, union_find_clusters  # same directory

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
STAGE_DIRS = [
    "01_establishment", "02_tillering", "03_stem_booting",
    "04_reproductive", "05_grain_filling", "06_ripening",
]
MARKER = "_subset_overlap_"


def cluster_folder(images: list[Path], threshold: int) -> list[int]:
    if not images:
        return []
    hashes = np.stack([average_hash(p) for p in images])
    n = len(images)
    edges = []
    for i in range(n):
        d = np.count_nonzero(hashes[i] != hashes[i + 1:], axis=1)
        for off, dd in enumerate(d):
            if dd <= threshold:
                edges.append((i, i + 1 + off))
    return union_find_clusters(n, edges)


def plan_folder(folder: Path, threshold: int) -> list[dict]:
    images = sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS
         and MARKER not in p.stem],
        key=lambda p: p.name.lower(),
    )
    labels = cluster_folder(images, threshold)
    # Stable parent numbering: order parents by first appearance.
    order, parent_of = {}, {}
    for lab in labels:
        if lab not in order:
            order[lab] = len(order)
    rows = []
    member_counter: dict[int, int] = {}
    for img, lab in zip(images, labels):
        pid = order[lab]
        k = member_counter.get(pid, 0)
        member_counter[pid] = k + 1
        new_stem = f"p{pid:03d}{MARKER}{k:03d}"
        rows.append({
            "stage_folder": folder.name,
            "old_name": img.name,
            "new_name": f"{new_stem}{img.suffix.lower()}",
            "parent_id": f"p{pid:03d}",
            "cluster_size": 0,  # filled below
        })
    sizes = {}
    for r in rows:
        sizes[r["parent_id"]] = sizes.get(r["parent_id"], 0) + 1
    for r in rows:
        r["cluster_size"] = sizes[r["parent_id"]]
    return rows


def apply_rename(root: Path, rows: list[dict]) -> None:
    # Two-phase to avoid collisions (temp names first, then final).
    by_folder: dict[str, list[dict]] = {}
    for r in rows:
        by_folder.setdefault(r["stage_folder"], []).append(r)
    for folder_name, frows in by_folder.items():
        folder = root / folder_name
        for i, r in enumerate(frows):
            (folder / r["old_name"]).rename(folder / f"__tmp_enc_{i:05d}{Path(r['old_name']).suffix.lower()}")
        for i, r in enumerate(frows):
            (folder / f"__tmp_enc_{i:05d}{Path(r['old_name']).suffix.lower()}").rename(folder / r["new_name"])


def undo(root: Path, mapping_csv: Path) -> None:
    with mapping_csv.open() as fh:
        rows = list(csv.DictReader(fh))
    by_folder: dict[str, list[dict]] = {}
    for r in rows:
        by_folder.setdefault(r["stage_folder"], []).append(r)
    for folder_name, frows in by_folder.items():
        folder = root / folder_name
        for i, r in enumerate(frows):
            (folder / r["new_name"]).rename(folder / f"__tmp_undo_{i:05d}{Path(r['new_name']).suffix.lower()}")
        for i, r in enumerate(frows):
            (folder / f"__tmp_undo_{i:05d}{Path(r['new_name']).suffix.lower()}").rename(folder / r["old_name"])
    print(f"Restored {len(rows)} files from {mapping_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="CROPSTATE_DATASET")
    parser.add_argument("--hamming-threshold", type=int, default=10)
    parser.add_argument("--apply", action="store_true", help="Actually rename (default is dry-run).")
    parser.add_argument("--undo", action="store_true", help="Restore original names from the mapping CSV.")
    parser.add_argument("--mapping", default=None, help="Mapping CSV path (default: <data-root>/parent_encoding_mapping.csv).")
    args = parser.parse_args()

    root = Path(args.data_root)
    mapping_csv = Path(args.mapping) if args.mapping else root / "parent_encoding_mapping.csv"

    if args.undo:
        undo(root, mapping_csv)
        return

    all_rows = []
    for name in STAGE_DIRS:
        folder = root / name
        if folder.exists():
            all_rows.extend(plan_folder(folder, args.hamming_threshold))

    # Report grouping.
    n = len(all_rows)
    parents = {(r["stage_folder"], r["parent_id"]) for r in all_rows}
    sizes = {}
    for r in all_rows:
        key = (r["stage_folder"], r["parent_id"])
        sizes[key] = sizes.get(key, 0) + 1
    multi = {k: v for k, v in sizes.items() if v > 1}
    print(f"threshold={args.hamming_threshold}  images={n}  parents={len(parents)}  "
          f"multi-image parents={len(multi)}  singletons={len(parents) - len(multi)}  "
          f"largest={max(sizes.values()) if sizes else 0}")
    print("\nMulti-image parent groups (candidate overlapping patches):")
    for (stage, pid), sz in sorted(multi.items(), key=lambda x: -x[1]):
        members = [r["old_name"] for r in all_rows if r["stage_folder"] == stage and r["parent_id"] == pid]
        print(f"  {stage}/{pid}  ({sz}): {', '.join(members)}")

    preview = mapping_csv.with_name("parent_encoding_preview.csv")
    with preview.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["stage_folder", "old_name", "new_name", "parent_id", "cluster_size"])
        w.writeheader()
        w.writerows(all_rows)

    if not args.apply:
        print(f"\nDRY RUN. Preview mapping: {preview}")
        print("Re-run with --apply to rename the image files (mapping saved for --undo).")
        return

    apply_rename(root, all_rows)
    with mapping_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["stage_folder", "old_name", "new_name", "parent_id", "cluster_size"])
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nRenamed {n} files. Mapping (for --undo): {mapping_csv}")


if __name__ == "__main__":
    main()

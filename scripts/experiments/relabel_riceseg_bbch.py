"""Derive 6 CROPSTATE BBCH stage labels from the RiceSEG dataset.

RiceSEG (https://huggingface.co/datasets/PheniX-Lab/RiceSEG, MIT) ships per-pixel
segmentation masks with six classes and NO growth-stage label (stage is only in the
paper's tables). This turns it into a stage-classification manifest by reading the
visible morphology each mask encodes.

Layout this script expects (as downloaded):
  <dataset-root>/
    class_pixel_counts.xlsx     # per-mask pixel count of class_0..class_5
    crop_mapping.xlsx           # crop -> original image (unused; parent comes from stem)
    segmentation/<country>/<province>/{label,rgb}/<stem>.{png,jpg}

Class order (class_0..class_5): background, green vegetation, senescent vegetation,
panicle, weeds, duckweed  (verified: class_1/green dominates; class_4/5 rare).

Per-image features -> BBCH macro-stage:
    canopy         = green + senescent + panicle
    cover          = canopy / (all - weeds - duckweed)     # crop fill of the frame
    panicle_frac   = panicle / canopy                       # panicle emerged? (BBCH 51 boundary)
    senescent_frac = senescent / (green + senescent)        # yellowing (BBCH 71+ boundary)

    panicle_frac >= t : reproductive (sf<slo) / grain_filling (sf<shi) / ripening   [BBCH 50-92]
    else              : establishment (cover<clo) / tillering (cover<chi) / stem_booting [BBCH 00-49]

Defaults are calibrated to this dataset's real distribution (panicle present in ~18%
of images; cover tertiles 0.31/0.51 among pre-panicle images; senescent thresholds
0.05/0.20 aligned to BBCH milk/dough/maturity — NOT forced to balance classes, so the
real vegetative/reproductive skew is preserved and left to the training sampler).

The pre-panicle 3-way split (establishment/tillering/stem_booting) rests on canopy
cover alone, which BBCH really separates by tiller count / stem elongation — so those
rows carry review_flag=1. Panicle-based late stages are labelled confidently.

Output: a manifest for scripts/train_vision.py (image_path relative to --dataset-root,
macro_stage, parent_image_id, field_id, split=unassigned + features + review_flag).

Deps: pandas, openpyxl, numpy. Torch-free. Reads the xlsx, not the mask pixels.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

STAGE_NAMES = [
    "establishment", "tillering", "stem_booting",
    "reproductive", "grain_filling", "ripening",
]
STAGE_DIRS = {s: f"{i + 1:02d}_{s}" for i, s in enumerate(STAGE_NAMES)}
CLASS = {"background": 0, "green": 1, "senescent": 2, "panicle": 3, "weeds": 4, "duckweed": 5}


def parent_stem(stem: str) -> str:
    return stem.split("_subset_overlap", 1)[0]


def index_rgb_by_stem(seg_root: Path) -> dict[str, Path]:
    """Map every RGB crop's stem -> its path (stems are globally unique in RiceSEG)."""
    out: dict[str, Path] = {}
    for p in seg_root.rglob("*.jpg"):
        if p.parent.name == "rgb":
            out.setdefault(p.stem, p)
    return out


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    c = {name: df[f"class_{idx}"].to_numpy(dtype=np.int64) for name, idx in CLASS.items()}
    total = np.sum([c[n] for n in CLASS], axis=0)
    canopy = c["green"] + c["senescent"] + c["panicle"]
    frame = np.clip(total - c["weeds"] - c["duckweed"], 1, None)
    veg = np.clip(c["green"] + c["senescent"], 1, None)
    out = df.copy()
    out["cover"] = canopy / frame
    out["panicle_frac"] = c["panicle"] / np.clip(canopy, 1, None)
    out["senescent_frac"] = c["senescent"] / veg
    out["canopy_px"] = canopy
    return out


def classify_row(pf: float, sf: float, cover: float, t) -> tuple[str, bool]:
    if pf >= t.panicle_min:
        if sf < t.senescent_low:
            stage = "reproductive"
        elif sf < t.senescent_high:
            stage = "grain_filling"
        else:
            stage = "ripening"
        review = abs(sf - t.senescent_low) < t.margin or abs(sf - t.senescent_high) < t.margin
    else:
        if cover < t.cover_low:
            stage = "establishment"
        elif cover < t.cover_high:
            stage = "tillering"
        else:
            stage = "stem_booting"
        # Cover is a weak proxy for the pre-panicle split; flag boundary cases (near a
        # cover cut, or a few panicle pixels just below the emergence threshold) so
        # review_flag prioritises the genuinely ambiguous rows, not every early image.
        review = (abs(cover - t.cover_low) < t.margin or abs(cover - t.cover_high) < t.margin
                  or pf > 0)
    return stage, review


class Thresholds:
    def __init__(self, a):
        self.panicle_min = a.panicle_min
        self.senescent_low = a.senescent_low
        self.senescent_high = a.senescent_high
        self.cover_low = a.cover_low
        self.cover_high = a.cover_high
        self.margin = a.margin


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-root", default="CROPSTATE_DATASET")
    p.add_argument("--pixel-counts", default=None, help="Override path to class_pixel_counts.xlsx.")
    p.add_argument("--output", default="data/riceseg_stage_manifest.csv")
    p.add_argument("--link-root", help="If set, symlink RGB crops into <link-root>/0N_stage/ folders.")
    p.add_argument("--min-canopy-frac", type=float, default=0.02,
                   help="Drop near-empty crops whose cover is below this (mostly water/background).")
    # Thresholds (calibrated to this dataset; see module docstring).
    p.add_argument("--panicle-min", type=float, default=0.02)
    p.add_argument("--senescent-low", type=float, default=0.05)
    p.add_argument("--senescent-high", type=float, default=0.20)
    p.add_argument("--cover-low", type=float, default=0.31)
    p.add_argument("--cover-high", type=float, default=0.51)
    p.add_argument("--margin", type=float, default=0.05)
    args = p.parse_args()

    root = Path(args.dataset_root)
    seg_root = root / "segmentation"
    pixel_counts = Path(args.pixel_counts) if args.pixel_counts else root / "class_pixel_counts.xlsx"
    if not pixel_counts.exists():
        raise SystemExit(f"Missing {pixel_counts} — is --dataset-root the RiceSEG folder?")
    if not seg_root.exists():
        raise SystemExit(f"Missing {seg_root}/ — expected RiceSEG segmentation/<country>/<prov>/rgb/")

    df = pd.read_excel(pixel_counts)
    missing = [f"class_{i}" for i in range(6) if f"class_{i}" not in df.columns]
    if missing:
        raise SystemExit(f"{pixel_counts.name} missing columns {missing}; got {list(df.columns)}")
    df["stem"] = df["image"].astype(str).str.replace("\\", "/", regex=False).map(lambda s: Path(s).stem)

    rgb_by_stem = index_rgb_by_stem(seg_root)
    df["rgb"] = df["stem"].map(rgb_by_stem)
    n_total = len(df)
    df = df[df["rgb"].notna()].copy()
    n_unresolved = n_total - len(df)

    df = compute_features(df)
    n_before = len(df)
    df = df[df["cover"] >= args.min_canopy_frac].copy()
    n_empty = n_before - len(df)

    t = Thresholds(args)
    stages, reviews = [], []
    for pf, sf, cover in zip(df["panicle_frac"], df["senescent_frac"], df["cover"]):
        stage, review = classify_row(float(pf), float(sf), float(cover), t)
        stages.append(stage)
        reviews.append(int(review))
    df["macro_stage"] = stages
    df["review_flag"] = reviews

    rows = []
    for r in df.itertuples(index=False):
        rgb = Path(r.rgb)
        # province taken from the real on-disk path (xlsx province can disagree, e.g. Japan).
        parts = rgb.relative_to(seg_root).parts  # <country>/<province>/rgb/<file>
        country, province = parts[0], (parts[1] if len(parts) > 3 else "")
        parent = parent_stem(rgb.stem)
        rows.append({
            "image_id": rgb.stem,
            "image_path": rgb.relative_to(root).as_posix(),
            "macro_stage": r.macro_stage,
            "parent_image_id": f"{country}/{province}/{parent}",
            "field_id": f"riceseg:{country}/{province}/{parent}",
            "capture_session": f"{country}/{province}",
            "country": country,
            "cover": round(float(r.cover), 4),
            "panicle_frac": round(float(r.panicle_frac), 4),
            "senescent_frac": round(float(r.senescent_frac), 4),
            "review_flag": int(r.review_flag),
            "source": "riceseg",
            "split": "unassigned",
        })

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest = pd.DataFrame(rows)
    manifest.to_csv(out, index=False)

    print(f"RiceSEG rows: {n_total} | unresolved rgb: {n_unresolved} | dropped near-empty: {n_empty} "
          f"| labelled: {len(manifest)}")
    print(f"Wrote {out}\n")
    print("Stage distribution (review-flagged in parentheses):")
    dist = manifest["macro_stage"].value_counts()
    rev = manifest[manifest.review_flag == 1]["macro_stage"].value_counts()
    for s in STAGE_NAMES:
        print(f"  {s:14s} {int(dist.get(s, 0)):5d}  ({int(rev.get(s, 0))} to review)")
    print(f"  parents (leak-free groups): {manifest['parent_image_id'].nunique()}")
    print(f"  flagged for review: {int(manifest.review_flag.sum())}/{len(manifest)}")

    if args.link_root:
        link_root = Path(args.link_root)
        for r in rows:
            d = link_root / STAGE_DIRS[r["macro_stage"]]
            d.mkdir(parents=True, exist_ok=True)
            dst = d / Path(r["image_path"]).name
            if not dst.exists():
                dst.symlink_to((root / r["image_path"]).resolve())
        print(f"\nSymlinked RGB crops into stage folders under {link_root}")


if __name__ == "__main__":
    main()

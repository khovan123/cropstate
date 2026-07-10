from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def _average_hash(path: str | Path, size: int = 8) -> np.ndarray:
    """64-bit perceptual average-hash (same definition as leakage_audit)."""
    from PIL import Image

    image = Image.open(path).convert("L").resize((size, size), Image.BILINEAR)
    pixels = np.asarray(image, dtype=np.float64)
    return (pixels > pixels.mean()).flatten()


def _union_find(n: int, edges: list[tuple[int, int]]) -> list[int]:
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    return [find(i) for i in range(n)]


def assign_leakfree_groups(
    df: pd.DataFrame,
    data_root: str | Path,
    hamming_threshold: int = 10,
    group_col: str = "parent_image_id",
    out_col: str = "leak_group",
) -> pd.DataFrame:
    """Assign a split-group id that is leak-free w.r.t. near-duplicate images.

    Two images share a group when EITHER (a) they carry the same ``group_col``
    value (keeps a parent's overlapping patches together) OR (b) their perceptual
    average-hashes are within ``hamming_threshold`` — computed GLOBALLY across all
    stage folders, so cross-stage near-duplicates cannot land in different splits.

    Splitting on the returned ``out_col`` guarantees ``leakage_audit`` reports zero
    test/val near-duplicate leak pairs at the same threshold.
    """
    root = Path(data_root)
    out = df.reset_index(drop=True).copy()
    n = len(out)
    if n == 0:
        out[out_col] = []
        return out

    edges: list[tuple[int, int]] = []
    if hamming_threshold is not None and hamming_threshold >= 0:
        hashes = np.stack([_average_hash(root / str(p)) for p in out["image_path"]])
        for i in range(n):
            dists = np.count_nonzero(hashes[i] != hashes[i + 1:], axis=1)
            for offset, dist in enumerate(dists):
                if dist <= hamming_threshold:
                    edges.append((i, i + 1 + offset))

    if group_col in out.columns:
        for _, positions in out.groupby(group_col).indices.items():
            anchor = int(positions[0])
            for pos in positions[1:]:
                edges.append((anchor, int(pos)))

    components = _union_find(n, edges)
    out[out_col] = [f"g{component:05d}" for component in components]
    return out


def grouped_train_val_test_split(
    df: pd.DataFrame,
    group_col: str = "field_id",
    test_size: float = 0.2,
    val_size: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by field/parent/session so related images never cross partitions."""
    if group_col not in df.columns:
        raise ValueError(f"Missing grouping column: {group_col}")
    if df[group_col].isna().any():
        raise ValueError(f"Grouping column {group_col} contains missing values")

    outer = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_val_idx, test_idx = next(outer.split(df, groups=df[group_col]))
    train_val = df.iloc[train_val_idx].copy()
    test = df.iloc[test_idx].copy()

    adjusted_val = val_size / (1.0 - test_size)
    inner = GroupShuffleSplit(n_splits=1, test_size=adjusted_val, random_state=seed + 1)
    train_idx, val_idx = next(inner.split(train_val, groups=train_val[group_col]))
    train = train_val.iloc[train_idx].copy()
    val = train_val.iloc[val_idx].copy()

    train["split"] = "train"
    val["split"] = "validation"
    test["split"] = "test"
    return train, val, test


def assert_no_group_leakage(df: pd.DataFrame, group_cols: list[str]) -> None:
    if "split" not in df.columns:
        raise ValueError("Manifest must contain a split column")
    for col in group_cols:
        if col not in df.columns:
            continue
        counts = df.groupby(col)["split"].nunique(dropna=False)
        leaking = counts[counts > 1]
        if not leaking.empty:
            raise AssertionError(f"Leakage detected in {col}: {leaking.index.tolist()[:10]}")

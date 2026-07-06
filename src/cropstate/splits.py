from __future__ import annotations

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


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

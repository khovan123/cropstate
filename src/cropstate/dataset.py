from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from .constants import STAGE_ALIASES, STAGE_TO_ID


def canonical_stage_label(value: object) -> str:
    slug = str(value).strip().lower().replace("/", "_").replace("-", "_")
    slug = "_".join(part for part in slug.split() if part)
    slug = "_".join(part for part in slug.split("_") if part)
    try:
        return STAGE_ALIASES[slug]
    except KeyError as exc:
        raise KeyError(f"Unknown macro_stage label: {value!r}") from exc


class RiceStageDataset(Dataset):
    """Image classification dataset backed by a metadata manifest."""

    def __init__(self, manifest: pd.DataFrame, root: str | Path, transform: Callable | None = None):
        self.df = manifest.reset_index(drop=True).copy()
        self.root = Path(root)
        self.transform = transform
        required = {"image_path", "macro_stage"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"Manifest missing columns: {sorted(missing)}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        image_path = self.root / str(row["image_path"])
        if not image_path.exists():
            raise FileNotFoundError(image_path)
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = STAGE_TO_ID[canonical_stage_label(row["macro_stage"])]
        return image, torch.tensor(label, dtype=torch.long), str(row.get("image_id", index))

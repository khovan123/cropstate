"""Export per-image logits/probabilities for saved checkpoints on val+test splits.

Feeds the calibration, temporal-fusion, and retrieval-gating experiments so they
never need to re-run the network. Raw logits are kept (not just softmax) because
temperature scaling is fitted on validation logits.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from cropstate.constants import STAGE_NAMES
from cropstate.dataset import RiceStageDataset, canonical_stage_label
from cropstate.vision import build_classifier


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"], ckpt.get("model_name", "resnet18"), int(ckpt.get("image_size", 224))
    return ckpt, "resnet18", 224


def eval_transform(image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def export_split(model, df, split, data_root, image_size, device):
    subset = df[df.split == split].copy()
    if subset.empty:
        return None
    ds = RiceStageDataset(subset, data_root, eval_transform(image_size))
    loader = DataLoader(ds, batch_size=32, shuffle=False)
    logits_all, labels_all, ids_all = [], [], []
    with torch.no_grad():
        for images, labels, ids in loader:
            logits = model(images.to(device)).cpu().numpy()
            logits_all.append(logits)
            labels_all.extend(labels.numpy().tolist())
            ids_all.extend(list(ids))
    logits = np.concatenate(logits_all, axis=0)
    return {
        "logits": logits,
        "labels": np.array(labels_all, dtype=int),
        "image_ids": ids_all,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="CROPSTATE_RESULTS/vision_final/manifest.csv")
    parser.add_argument("--data-root", default="CROPSTATE_DATASET")
    parser.add_argument("--checkpoint", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--output-dir", default="CROPSTATE_RESULTS/novelty/logits")
    args = parser.parse_args()

    device = torch.device("cpu") if os.environ.get("CROPSTATE_FORCE_CPU") == "1" \
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(args.manifest)
    df["macro_stage"] = df["macro_stage"].map(canonical_stage_label)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    index = {}
    for entry in args.checkpoint:
        name, _, path = entry.partition("=")
        state_dict, model_name, image_size = load_checkpoint(path, device)
        model = build_classifier(model_name, num_classes=len(STAGE_NAMES), pretrained=False).to(device)
        model.load_state_dict(state_dict)
        model.eval()
        payload = {}
        for split in ("validation", "test"):
            data = export_split(model, df, split, args.data_root, image_size, device)
            if data is not None:
                payload[split] = data
        npz_path = out_dir / f"{name}.npz"
        np.savez(
            npz_path,
            **{f"{split}_{key}": value for split, data in payload.items()
               for key, value in data.items() if key != "image_ids"},
            **{f"{split}_image_ids": np.array(data["image_ids"]) for split, data in payload.items()},
        )
        index[name] = {"model_name": model_name, "image_size": image_size, "path": str(npz_path)}
        print(f"[export] {name}: " + ", ".join(f"{s}={payload[s]['logits'].shape[0]}" for s in payload))

    (out_dir / "index.json").write_text(json.dumps(index, indent=2))


if __name__ == "__main__":
    main()

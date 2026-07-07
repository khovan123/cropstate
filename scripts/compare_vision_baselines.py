from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cropstate.constants import STAGE_TO_ID
from cropstate.dataset import canonical_stage_label
from cropstate.statistics import holm_adjust, paired_bootstrap_ci, paired_wilcoxon
from predict_image import load_model, predict_with_model


def per_image_correctness(checkpoint: Path, manifest: pd.DataFrame, data_root: Path) -> dict[str, int]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, image_size = load_model(checkpoint, device)
    correctness = {}
    for row in manifest.itertuples():
        result = predict_with_model(model, image_size, data_root / row.image_path, device)
        true_id = STAGE_TO_ID[canonical_stage_label(row.macro_stage)]
        correctness[row.image_path] = int(STAGE_TO_ID[result["predicted_stage"]] == true_id)
    return correctness


def majority_correctness(manifest: pd.DataFrame, majority_stage: str) -> dict[str, int]:
    return {
        row.image_path: int(canonical_stage_label(row.macro_stage) == majority_stage)
        for row in manifest.itertuples()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired comparison of vision baselines on one shared test split.")
    parser.add_argument("--manifest", required=True, help="Manifest CSV whose 'test' rows define the shared split.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument(
        "--checkpoint", action="append", required=True, metavar="NAME=PATH",
        help="Repeatable. e.g. --checkpoint resnet18=results/vision_final/best_checkpoint.pt",
    )
    parser.add_argument("--output", default="results/vision_baseline_comparison.json")
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    test_rows = manifest[manifest.split == "test"].reset_index(drop=True)
    data_root = Path(args.data_root)

    train_rows = manifest[manifest.split == "train"]
    majority_stage = train_rows["macro_stage"].map(canonical_stage_label).value_counts().idxmax()

    correctness_by_model: dict[str, dict[str, int]] = {"majority_class": majority_correctness(test_rows, majority_stage)}
    for entry in args.checkpoint:
        name, _, path = entry.partition("=")
        correctness_by_model[name] = per_image_correctness(Path(path), test_rows, data_root)

    image_paths = test_rows["image_path"].tolist()
    vectors = {
        name: np.array([correctness[image_path] for image_path in image_paths], dtype=float)
        for name, correctness in correctness_by_model.items()
    }

    pairs = list(combinations(vectors, 2))
    p_values = []
    pairwise = []
    for name_a, name_b in pairs:
        a, b = vectors[name_a], vectors[name_b]
        mean_diff, ci = paired_bootstrap_ci(a, b)
        try:
            wilcoxon_result = paired_wilcoxon(a, b)
        except ValueError:
            wilcoxon_result = {"statistic": None, "p_value": 1.0}
        p_values.append(wilcoxon_result["p_value"])
        pairwise.append({
            "a": name_a, "b": name_b,
            "accuracy_a": float(a.mean()), "accuracy_b": float(b.mean()),
            "mean_difference": mean_diff, "bootstrap_ci_95": list(ci),
            "wilcoxon_p_value": wilcoxon_result["p_value"],
        })
    adjusted = holm_adjust(p_values)
    for row, adjusted_p in zip(pairwise, adjusted):
        row["holm_adjusted_p_value"] = adjusted_p

    accuracy_ci = {}
    for name, vector in vectors.items():
        mean_accuracy, ci = paired_bootstrap_ci(vector, np.zeros_like(vector))
        accuracy_ci[name] = {"accuracy": mean_accuracy, "bootstrap_ci_95": list(ci)}

    payload = {
        "test_set_size": len(image_paths),
        "accuracy": accuracy_ci,
        "per_image_correctness": {name: vector.astype(int).tolist() for name, vector in vectors.items()},
        "image_paths": image_paths,
        "pairwise_comparisons": pairwise,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

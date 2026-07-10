"""Temperature scaling + calibration under imbalance/ordinal structure (Tier A#3).

The paper reports raw ECE=0.250 and defers temperature scaling to future work.
This actually fits it on validation logits, applies it to the test split, and
reports ECE/Brier/NLL before and after — plus a confidence-thresholded abstention
policy that trades coverage for selective accuracy. Accuracy/argmax are unchanged
by temperature scaling; only confidence calibration moves.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

from cropstate.metrics import expected_calibration_error, multiclass_brier, vision_metrics


def softmax_np(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Fit scalar T>0 minimizing validation NLL (torch-free; avoids the CUDA probe)."""
    def objective(log_t):
        t = np.exp(log_t)
        probs = softmax_np(logits / t)
        return float(-np.mean(np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1.0))))

    result = minimize_scalar(objective, bounds=(np.log(0.05), np.log(10.0)), method="bounded")
    return float(np.exp(result.x))


def nll(probs: np.ndarray, labels: np.ndarray) -> float:
    return float(-np.mean(np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1.0))))


def calib_block(logits, labels):
    probs = softmax_np(logits)
    preds = probs.argmax(1)
    vm = vision_metrics(labels, preds)
    return {
        "accuracy": vm["accuracy"], "macro_f1": vm["macro_f1"], "masd": vm["masd"],
        "ece": expected_calibration_error(probs, labels),
        "brier": multiclass_brier(probs, labels),
        "nll": nll(probs, labels),
        "mean_confidence": float(probs.max(1).mean()),
    }


def abstention_curve(probs, labels, thresholds):
    preds = probs.argmax(1)
    conf = probs.max(1)
    correct = (preds == labels)
    rows = []
    for t in thresholds:
        keep = conf >= t
        cov = float(keep.mean())
        acc = float(correct[keep].mean()) if keep.any() else float("nan")
        rows.append({"threshold": round(float(t), 3), "coverage": cov, "selective_accuracy": acc})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logits-index", default="CROPSTATE_RESULTS/novelty/logits/index.json")
    parser.add_argument("--output", default="CROPSTATE_RESULTS/novelty/calibration.json")
    args = parser.parse_args()

    index = json.loads(Path(args.logits_index).read_text())
    results = {}
    for name, info in index.items():
        data = np.load(info["path"], allow_pickle=True)
        if "validation_logits" not in data or "test_logits" not in data:
            continue
        val_logits = data["validation_logits"].astype(np.float64)
        val_labels = data["validation_labels"].astype(int)
        test_logits = data["test_logits"].astype(np.float64)
        test_labels = data["test_labels"].astype(int)

        temperature = fit_temperature(val_logits, val_labels)
        scaled_test = test_logits / temperature

        before = calib_block(test_logits, test_labels)
        after = calib_block(scaled_test, test_labels)
        thresholds = np.linspace(0.3, 0.95, 14)
        results[name] = {
            "temperature": temperature,
            "test_before": before,
            "test_after": after,
            "ece_reduction": before["ece"] - after["ece"],
            "brier_reduction": before["brier"] - after["brier"],
            "abstention_curve_after": abstention_curve(softmax_np(scaled_test), test_labels, thresholds),
        }
        print(f"[{name}] T={temperature:.3f}  ECE {before['ece']:.3f} -> {after['ece']:.3f}   "
              f"Brier {before['brier']:.3f} -> {after['brier']:.3f}   NLL {before['nll']:.3f} -> {after['nll']:.3f}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split

from cropstate.constants import STAGE_BBCH_RANGES, STAGE_NAMES, STAGE_TO_ID
from cropstate.dataset import RiceStageDataset, canonical_stage_label
from cropstate.metrics import expected_calibration_error, multiclass_brier, vision_metrics
from cropstate.splits import assert_no_group_leakage
from cropstate.vision import build_classifier

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
STAGE_DIR_RE = re.compile(r"^\s*(?P<number>0?[1-6])[\s_-]+(?P<name>.+?)\s*$")


def normalize_stage_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def stage_from_folder_name(folder_name: str) -> str | None:
    match = STAGE_DIR_RE.match(folder_name)
    if not match:
        return None
    index = int(match.group("number")) - 1
    if not 0 <= index < len(STAGE_NAMES):
        return None
    parsed_name = normalize_stage_name(match.group("name"))
    expected = STAGE_NAMES[index]
    if parsed_name and parsed_name != expected:
        aliases = {
            "stem_booting": {"stem_booting", "stem", "booting", "stem_elongation_booting"},
            "grain_filling": {"grain_filling", "grain_development"},
        }
        if parsed_name not in aliases.get(expected, {expected}):
            raise ValueError(f"Stage folder {folder_name!r} maps to {expected!r}, got slug {parsed_name!r}")
    return expected


def parent_image_id(path: Path) -> str:
    return path.stem.split("_subset_overlap", 1)[0]


def build_manifest_from_stage_folders(data_root: str | Path) -> pd.DataFrame:
    """Create an image manifest from paper-aligned stage folders under data_root."""
    root = Path(data_root)
    rows = []
    for stage_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        stage = stage_from_folder_name(stage_dir.name)
        if stage is None:
            continue
        for image_path in sorted(p for p in stage_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS):
            relative = image_path.relative_to(root)
            parent = parent_image_id(image_path)
            subdirs = relative.parts[1:-1]
            capture_session = "/".join(subdirs) if subdirs else "unknown"
            rows.append({
                "image_id": image_path.stem,
                "image_path": relative.as_posix(),
                "parent_image_id": f"{stage}:{capture_session}:{parent}",
                "field_id": f"{stage}:{capture_session}:{parent}",
                "capture_session": capture_session,
                "capture_date": "",
                "season": "unknown",
                "variety": "unknown",
                "days_after_sowing": "",
                "bbch_code": STAGE_BBCH_RANGES[stage],
                "macro_stage": stage,
                "source": "stage_folder",
                "license": "user_provided",
                "annotator_1": "",
                "annotator_2": "",
                "adjudicated_label": stage,
                "split": "unassigned",
            })
    if not rows:
        raise ValueError(f"No stage images found under {root}")
    return pd.DataFrame(rows)


def grouped_stratified_split(
    df: pd.DataFrame,
    group_col: str = "parent_image_id",
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    """Split by parent/group while approximately preserving stage proportions."""
    groups = df.groupby(group_col, as_index=False).agg(macro_stage=("macro_stage", "first"))
    stage_counts = groups["macro_stage"].value_counts()
    stratify_outer = groups["macro_stage"] if stage_counts.min() >= 2 else None
    try:
        train_val_groups, test_groups = train_test_split(
            groups,
            test_size=test_size,
            random_state=seed,
            stratify=stratify_outer,
        )
    except ValueError:
        train_val_groups, test_groups = train_test_split(
            groups,
            test_size=test_size,
            random_state=seed,
            stratify=None,
        )

    train_val_counts = train_val_groups["macro_stage"].value_counts()
    adjusted_val = val_size / (1.0 - test_size)
    stratify_inner = train_val_groups["macro_stage"] if train_val_counts.min() >= 2 else None
    try:
        train_groups, val_groups = train_test_split(
            train_val_groups,
            test_size=adjusted_val,
            random_state=seed + 1,
            stratify=stratify_inner,
        )
    except ValueError:
        train_groups, val_groups = train_test_split(
            train_val_groups,
            test_size=adjusted_val,
            random_state=seed + 1,
            stratify=None,
        )

    split_by_group = {
        **{group: "train" for group in train_groups[group_col]},
        **{group: "validation" for group in val_groups[group_col]},
        **{group: "test" for group in test_groups[group_col]},
    }
    out = df.copy()
    out["split"] = out[group_col].map(split_by_group)
    assert_no_group_leakage(out, [group_col, "parent_image_id"])
    return out


def load_config(path: str | Path | None) -> dict:
    if path is None:
        return {}
    with Path(path).open() as handle:
        return yaml.safe_load(handle) or {}


def dataset_from_split(df: pd.DataFrame, split: str, root: str | Path, transform):
    subset = df[df.split == split].copy()
    if subset.empty:
        raise ValueError(f"Split {split!r} is empty")
    return RiceStageDataset(subset, root, transform)


def optional_dataset_from_split(df: pd.DataFrame, split: str, root: str | Path, transform):
    subset = df[df.split == split].copy()
    if subset.empty:
        return None
    return RiceStageDataset(subset, root, transform)


def classifier_parameter_ids(model: nn.Module) -> set[int]:
    classifier = model.get_classifier() if hasattr(model, "get_classifier") else None
    if isinstance(classifier, nn.Module):
        return {id(param) for param in classifier.parameters()}
    head_names = ("classifier", "fc", "head")
    return {
        id(param)
        for name, param in model.named_parameters()
        if name.startswith(head_names) or any(f".{head_name}." in name for head_name in head_names)
    }


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    classifier_ids = classifier_parameter_ids(model)
    for param in model.parameters():
        param.requires_grad = trainable or id(param) in classifier_ids


def build_optimizer(model: nn.Module, learning_rate: float, backbone_learning_rate: float, weight_decay: float):
    classifier_ids = classifier_parameter_ids(model)
    head_params = []
    backbone_params = []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        if id(param) in classifier_ids:
            head_params.append(param)
        else:
            backbone_params.append(param)

    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": backbone_learning_rate})
    if head_params:
        param_groups.append({"params": head_params, "lr": learning_rate})
    if not param_groups:
        raise ValueError("No trainable parameters found")
    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)


def load_model_state_from_checkpoint(path: str | Path, device: torch.device) -> tuple[dict, dict]:
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        metadata = {key: value for key, value in checkpoint.items() if key != "model_state_dict"}
        return checkpoint["model_state_dict"], metadata
    return checkpoint, {}


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    losses, labels_all, preds_all, probs_all = [], [], [], []
    for images, labels, _ in tqdm(loader, leave=False):
        images, labels = images.to(device), labels.to(device)
        if training:
            optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        if training:
            loss.backward()
            optimizer.step()
        losses.append(loss.item())
        labels_all.extend(labels.detach().cpu().numpy())
        probabilities = torch.softmax(logits.detach(), dim=1)
        preds_all.extend(probabilities.argmax(1).cpu().numpy())
        probs_all.extend(probabilities.cpu().numpy())
    labels_array = np.array(labels_all)
    preds_array = np.array(preds_all)
    probs_array = np.array(probs_all)
    metrics = vision_metrics(labels_array, preds_array)
    metrics["loss"] = float(np.mean(losses))
    metrics["brier"] = multiclass_brier(probs_array, labels_array)
    metrics["ece"] = expected_calibration_error(probs_array, labels_array)
    metrics["per_class_recall"] = {
        stage: float(value)
        for stage, value in zip(
            STAGE_NAMES,
            recall_score(labels_array, preds_array, labels=list(range(len(STAGE_NAMES))), average=None, zero_division=0),
            strict=True,
        )
    }
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", help="CSV manifest. If omitted, build one from stage folders in --data-root.")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--config", default="configs/vision.yaml")
    parser.add_argument("--output", default="results/vision_final")
    parser.add_argument("--model")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--backbone-learning-rate", type=float)
    parser.add_argument("--freeze-backbone-epochs", type=int)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--split-group-column")
    parser.add_argument("--val-size", type=float)
    parser.add_argument("--test-size", type=float)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument("--resume-checkpoint", help="Continue fine-tuning from a previous best_checkpoint.pt or state_dict.")
    parser.add_argument("--write-manifest", default="manifest.csv")
    args = parser.parse_args()

    config = load_config(args.config)
    seed = args.seed if args.seed is not None else int(config.get("seed", 42))
    image_size = args.image_size if args.image_size is not None else int(config.get("image_size", 224))
    batch_size = args.batch_size if args.batch_size is not None else int(config.get("batch_size", 32))
    epochs = args.epochs if args.epochs is not None else int(config.get("epochs", 30))
    model_name = args.model or config.get("model_name", "resnet18")
    learning_rate = args.learning_rate if args.learning_rate is not None else float(config.get("learning_rate", 3e-4))
    backbone_learning_rate = (
        args.backbone_learning_rate
        if args.backbone_learning_rate is not None
        else float(config.get("backbone_learning_rate", learning_rate * 0.1))
    )
    freeze_backbone_epochs = (
        args.freeze_backbone_epochs
        if args.freeze_backbone_epochs is not None
        else int(config.get("freeze_backbone_epochs", 3))
    )
    weight_decay = args.weight_decay if args.weight_decay is not None else float(config.get("weight_decay", 1e-4))
    val_size = args.val_size if args.val_size is not None else float(config.get("validation_size", 0.15))
    test_size = args.test_size if args.test_size is not None else float(config.get("test_size", 0.15))
    pretrained = bool(config.get("pretrained", True)) and not args.no_pretrained
    split_group_column = args.split_group_column or config.get("split_group_column", "parent_image_id")

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    manifest_output = output / args.write_manifest
    for legacy_path in (output / "best_model.pt", output / "manifest_from_folders.csv"):
        if legacy_path.exists() and legacy_path != manifest_output:
            legacy_path.unlink()

    if args.manifest:
        df = pd.read_csv(args.manifest)
    else:
        df = build_manifest_from_stage_folders(args.data_root)

    df["macro_stage"] = df["macro_stage"].map(canonical_stage_label)
    if split_group_column not in df.columns:
        split_group_column = "parent_image_id"
    if "split" not in df.columns or df["split"].isna().any() or (df["split"] == "unassigned").any():
        df = grouped_stratified_split(
            df,
            group_col=split_group_column,
            test_size=test_size,
            val_size=val_size,
            seed=seed,
        )
    assert_no_group_leakage(df, [split_group_column, "parent_image_id"])
    df.to_csv(manifest_output, index=False)

    train_tf = transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15),
        transforms.RandomRotation(8),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train = dataset_from_split(df, "train", args.data_root, train_tf)
    val = dataset_from_split(df, "validation", args.data_root, eval_tf)
    test = optional_dataset_from_split(df, "test", args.data_root, eval_tf)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=2) if test is not None else None

    model = build_classifier(
        model_name,
        num_classes=len(STAGE_NAMES),
        pretrained=pretrained and not args.resume_checkpoint,
    ).to(device)
    resume_metadata = {}
    if args.resume_checkpoint:
        state_dict, resume_metadata = load_model_state_from_checkpoint(args.resume_checkpoint, device)
        checkpoint_model_name = resume_metadata.get("model_name")
        if checkpoint_model_name and checkpoint_model_name != model_name:
            raise ValueError(
                f"Checkpoint model_name={checkpoint_model_name!r} does not match requested model={model_name!r}"
            )
        model.load_state_dict(state_dict)
    if freeze_backbone_epochs > 0:
        set_backbone_trainable(model, trainable=False)
    if args.no_class_weights:
        criterion = nn.CrossEntropyLoss()
    else:
        class_counts = df[df.split == "train"]["macro_stage"].map(STAGE_TO_ID).value_counts().reindex(range(len(STAGE_NAMES)), fill_value=0)
        weights = class_counts.sum() / (len(STAGE_NAMES) * class_counts.clip(lower=1))
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights.to_numpy(dtype=np.float32), device=device))
    optimizer = build_optimizer(model, learning_rate, backbone_learning_rate, weight_decay)
    resume_validation = resume_metadata.get("validation") or {}
    best_f1 = float(resume_validation.get("macro_f1", -1.0) or -1.0)
    history = []
    history_path = output / "history.json"
    history_epoch_offset = 0
    if args.resume_checkpoint and history_path.exists():
        try:
            loaded_history = json.loads(history_path.read_text())
            if isinstance(loaded_history, list):
                history = loaded_history
                history_epoch_offset = max((int(item.get("epoch", 0)) for item in history), default=0)
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            history = []
            history_epoch_offset = 0

    for epoch in range(1, epochs + 1):
        if freeze_backbone_epochs > 0 and epoch == freeze_backbone_epochs + 1:
            set_backbone_trainable(model, trainable=True)
            optimizer = build_optimizer(model, learning_rate, backbone_learning_rate, weight_decay)
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, criterion, device)
        phase = "head_only" if epoch <= freeze_backbone_epochs else "full_finetune"
        global_epoch = history_epoch_offset + epoch
        history.append({
            "epoch": global_epoch,
            "run_epoch": epoch,
            "phase": phase,
            "train": train_metrics,
            "validation": val_metrics,
        })
        print(global_epoch, val_metrics)
        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_name": model_name,
                "num_classes": len(STAGE_NAMES),
                "class_names": STAGE_NAMES,
                "image_size": image_size,
                "best_epoch": global_epoch,
                "best_run_epoch": epoch,
                "phase": phase,
                "resume_checkpoint": args.resume_checkpoint,
                "resume_metadata": resume_metadata,
                "config": {
                    "seed": seed,
                    "learning_rate": learning_rate,
                    "backbone_learning_rate": backbone_learning_rate,
                    "freeze_backbone_epochs": freeze_backbone_epochs,
                    "weight_decay": weight_decay,
                    "batch_size": batch_size,
                    "epochs": epochs,
                    "pretrained": pretrained,
                    "split_group_column": split_group_column,
                    "validation_size": val_size,
                    "test_size": test_size,
                },
                "validation": val_metrics,
            }, output / "best_checkpoint.pt")
    if not (output / "best_checkpoint.pt").exists():
        raise FileNotFoundError(output / "best_checkpoint.pt")

    state_dict, best_metadata = load_model_state_from_checkpoint(output / "best_checkpoint.pt", device)
    model.load_state_dict(state_dict)
    if test_loader is None:
        test_metrics = {
            "error": "Split 'test' is empty; no independent test metrics were computed.",
            "best_checkpoint": best_metadata,
        }
    else:
        with torch.no_grad():
            test_metrics = run_epoch(model, test_loader, criterion, device)
        test_metrics["best_checkpoint"] = {
            "best_epoch": best_metadata.get("best_epoch"),
            "validation": best_metadata.get("validation"),
        }

    history_path.write_text(json.dumps(history, indent=2))
    (output / "class_counts.json").write_text(
        json.dumps(df.groupby(["split", "macro_stage"]).size().unstack(fill_value=0).to_dict(orient="index"), indent=2)
    )
    (output / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from cropstate.dataset import RiceStageDataset
from cropstate.vision import build_classifier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", default="results/stage_beliefs.jsonl")
    args = parser.parse_args()

    df = pd.read_csv(args.manifest)
    df = df[df.split == args.split]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        model_name = checkpoint.get("model_name", "resnet18")
        image_size = int(checkpoint.get("image_size", 224))
    else:
        state_dict = checkpoint
        model_name = "resnet18"
        image_size = 224
    tf = transforms.Compose([
        transforms.Resize((image_size, image_size)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    dataset = RiceStageDataset(df, args.data_root, tf)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    model = build_classifier(model_name, pretrained=False).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle, torch.no_grad():
        for images, labels, ids in loader:
            probs = torch.softmax(model(images.to(device)), dim=1).cpu().numpy()
            for image_id, label, probability in zip(ids, labels.numpy(), probs, strict=True):
                handle.write(json.dumps({
                    "image_id": image_id,
                    "ground_truth_stage": int(label),
                    "stage_belief": probability.tolist(),
                    "predicted_stage": int(probability.argmax()),
                }) + "\n")


if __name__ == "__main__":
    main()

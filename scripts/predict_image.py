from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from cropstate.constants import STAGE_DISPLAY_NAMES, STAGE_NAMES
from cropstate.vision import build_classifier


def load_checkpoint(path: str | Path, device: torch.device) -> tuple[dict, str, int]:
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model_name = checkpoint.get("model_name", "resnet18")
        image_size = int(checkpoint.get("image_size", 224))
        return checkpoint["model_state_dict"], model_name, image_size
    return checkpoint, "resnet18", 224


def build_transform(image_size: int):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def load_model(checkpoint_path: str | Path, device: torch.device) -> tuple[torch.nn.Module, int]:
    state_dict, model_name, image_size = load_checkpoint(checkpoint_path, device)
    model = build_classifier(model_name, num_classes=len(STAGE_NAMES), pretrained=False).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, image_size


def predict_with_model(model: torch.nn.Module, image_size: int, image_path: str | Path, device: torch.device) -> dict:
    image = Image.open(image_path).convert("RGB")
    tensor = build_transform(image_size)(image).unsqueeze(0).to(device)
    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1).squeeze(0).cpu()

    predicted_id = int(probabilities.argmax().item())
    predicted_stage = STAGE_NAMES[predicted_id]
    return {
        "image_path": str(image_path),
        "predicted_id": predicted_id,
        "predicted_stage": predicted_stage,
        "predicted_stage_display": STAGE_DISPLAY_NAMES[predicted_stage],
        "confidence": float(probabilities[predicted_id].item()),
        "stage_belief": {
            stage: float(probabilities[index].item())
            for index, stage in enumerate(STAGE_NAMES)
        },
    }


def predict_image(checkpoint_path: str | Path, image_path: str | Path) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, image_size = load_model(checkpoint_path, device)
    return predict_with_model(model, image_size, image_path, device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="results/vision_final/best_checkpoint.pt")
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", help="Optional JSON output path.")
    args = parser.parse_args()

    result = predict_image(args.checkpoint, args.image)
    print(json.dumps(result, indent=2))
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

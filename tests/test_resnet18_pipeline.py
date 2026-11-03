

"""
Single-image inference test for MelanSight ResNet-18.

Usage:
    python tests/test_resnet18_pipeline.py path/to/image.jpg

Optional:
    python tests/test_resnet18_pipeline.py path/to/image.jpg --experiment focal
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# Resolve project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.resnet18 import create_resnet18, CLASS_NAMES


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINTS = {
    "ce": "resnet18_ce",
    "focal": "resnet18_focal",
    "weighted_ce": "resnet18_weighted_ce",
}

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# --------------------------------------------------
# Load checkpoint
# --------------------------------------------------

def load_model(experiment="weighted_ce"):
    if experiment not in CHECKPOINTS:
        raise ValueError(
            f"Unknown experiment: {experiment}. "
            f"Choose from {list(CHECKPOINTS)}"
        )

    checkpoint_path = (
        PROJECT_ROOT
        / "checkpoints"
        / "resnet18"
        / CHECKPOINTS[experiment]
        / "best.pt"
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n\n"
            "Expected structure: checkpoints/resnet18/<experiment_dir>/best.pt\n"
            f"Available experiments: {list(CHECKPOINTS.keys())}\n\n"
            f"Please train ResNet-18 first or check the checkpoint path."
        )

    print(f"Loading checkpoint: {checkpoint_path}")

    model = create_resnet18(
        num_classes=len(CLASS_NAMES),
        pretrained=False
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=True
    )

    # Support both trainer checkpoints and raw state dictionaries.
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    return model


# --------------------------------------------------
# Image prediction
# --------------------------------------------------

def predict_image(model, image_path):
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image does not exist: {image_path}"
        )

    image = Image.open(image_path).convert("RGB")

    input_tensor = IMAGE_TRANSFORM(image)
    input_tensor = input_tensor.unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        logits = model(input_tensor)
        probabilities = F.softmax(logits, dim=1)[0]

    predicted_index = probabilities.argmax().item()

    return {
        "predicted_class": CLASS_NAMES[predicted_index],
        "confidence": probabilities[predicted_index].item(),
        "probabilities": {
            class_name: probabilities[i].item()
            for i, class_name in enumerate(CLASS_NAMES)
        }
    }


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Test MelanSight ResNet-18 on one image"
    )

    parser.add_argument(
        "image",
        type=Path,
        help="Path to the input skin-lesion image"
    )

    parser.add_argument(
        "--experiment",
        choices=list(CHECKPOINTS),
        default="focal",
        help="Which trained checkpoint to use"
    )

    args = parser.parse_args()

    print("=" * 55)
    print("MELANSIGHT - RESNET18 SINGLE IMAGE TEST")
    print("=" * 55)

    print(f"Device: {DEVICE}")
    print(f"Experiment: {args.experiment}")
    print(f"Image: {args.image}")

    model = load_model(args.experiment)
    result = predict_image(model, args.image)

    print("\nPrediction")
    print("-" * 55)

    print(f"Predicted class: {result['predicted_class']}")
    print(f"Confidence: {result['confidence'] * 100:.2f}%")

    print("\nClass probabilities")
    print("-" * 55)

    for class_name, probability in result["probabilities"].items():
        bar = "#" * int(probability * 30)

        print(
            f"{class_name:8s} "
            f"{probability * 100:6.2f}%  {bar}"
        )

    print("=" * 55)
    print("Inference completed successfully.")


if __name__ == "__main__":
    main()

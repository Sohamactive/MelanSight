"""
Single-image inference test for EfficientNet-B0 model.

Usage:
    python tests/test_efficientnet_b0_pipeline.py path/to/image.jpg

Optional:
    python tests/test_efficientnet_b0_pipeline.py path/to/image.jpg --experiment focal
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

from src.models.efficientnet_b0 import create_efficientnet_b0, CLASS_NAMES


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define available checkpoints
# Format: experiment_name: checkpoint_dir_name
CHECKPOINTS = {
    "ce": "ce_aug-false",
    "ce_aug": "ce_aug-true",
    "focal": "focal_aug-false",
    "focal_aug": "focal_aug-true",
    "weighted_ce": "weighted_ce_aug-false",
    "weighted_ce_aug": "weighted_ce_aug-true",
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
    """Load EfficientNet-B0 model from checkpoint.
    
    Args:
        experiment: Which checkpoint to use ('ce', 'focal', 'weighted_ce')
    
    Returns:
        model: Loaded EfficientNet-B0 model in eval mode
    """
    if experiment not in CHECKPOINTS:
        raise ValueError(
            f"Unknown experiment: {experiment}. "
            f"Choose from {list(CHECKPOINTS)}"
        )

    checkpoint_path = (
        PROJECT_ROOT
        / "checkpoint"  # Note: singular 'checkpoint' not 'checkpoints'
        / "efficientnet_b0"
        / CHECKPOINTS[experiment]
        / "best.pt"
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n\n"
            f"Expected structure: checkpoint/efficientnet_b0/<experiment_dir>/best.pt\n"
            f"Available experiments: {list(CHECKPOINTS.keys())}\n\n"
            f"Please train EfficientNet-B0 first or check the checkpoint path."
        )

    print(f"Loading checkpoint: {checkpoint_path}")

    model = create_efficientnet_b0(
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
    """Run inference on a single image.
    
    Args:
        model: Loaded EfficientNet-B0 model
        image_path: Path to input image
    
    Returns:
        dict with predicted_class, confidence, and all probabilities
    """
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
        description="Test EfficientNet-B0 on one skin lesion image"
    )

    parser.add_argument(
        "image",
        type=Path,
        help="Path to the input skin-lesion image"
    )

    parser.add_argument(
        "--experiment",
        choices=list(CHECKPOINTS),
        default="focal_aug",
        help="Which trained checkpoint to use (e.g., focal_aug for focal loss with augmentation)"
    )

    args = parser.parse_args()

    print("=" * 55)
    print("EFFICIENTNET-B0 SINGLE IMAGE TEST")
    print("=" * 55)

    print(f"Device: {DEVICE}")
    print(f"Experiment: {args.experiment}")
    print(f"Image: {args.image}")

    try:
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
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease ensure you have:")
        print("  1. Trained the EfficientNet-B0 model")
        print("  2. Saved checkpoints to checkpoint/efficientnet_b0/<experiment>/best.pt")
        print("\nExpected checkpoint structure:")
        print("  checkpoint/efficientnet_b0/ce_aug-false/best.pt")
        print("  checkpoint/efficientnet_b0/ce_aug-true/best.pt")
        print("  checkpoint/efficientnet_b0/focal_aug-false/best.pt")
        print("  checkpoint/efficientnet_b0/focal_aug-true/best.pt")
        sys.exit(1)
    
    except Exception as e:
        print(f"\nError during inference: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

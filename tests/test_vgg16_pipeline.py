"""
Single-image inference test for VGG-16 model.

Usage:
    python tests/test_vgg16_pipeline.py path/to/image.jpg
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

from src.models.vgg16 import create_vgg16, CLASS_NAMES


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "vgg16"
    / "focal_aug-true"
    / "best.pt"
)

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

def load_model():
    """Load the trained VGG-16 model from the single project checkpoint."""
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}\n\n"
            "Expected structure: checkpoints/vgg16/focal_aug-true/best.pt\n\n"
            f"Please train VGG-16 first or check the checkpoint path."
        )

    print(f"Loading checkpoint: {CHECKPOINT_PATH}")

    model = create_vgg16(
        num_classes=len(CLASS_NAMES),
        pretrained=False
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
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
        model: Loaded VGG-16 model
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
        description="Test VGG-16 on one skin lesion image"
    )

    parser.add_argument(
        "image",
        type=Path,
        help="Path to the input skin-lesion image"
    )

    args = parser.parse_args()

    print("=" * 55)
    print("VGG-16 SINGLE IMAGE TEST")
    print("=" * 55)

    print(f"Device: {DEVICE}")
    print("Checkpoint: focal_aug-true")
    print(f"Image: {args.image}")

    try:
        model = load_model()
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
        print("  1. Trained the VGG-16 model")
        print("  2. Saved the checkpoint to checkpoints/vgg16/focal_aug-true/best.pt")
        print("\nExpected checkpoint structure:")
        print("  checkpoints/vgg16/focal_aug-true/best.pt")
        sys.exit(1)
    
    except Exception as e:
        print(f"\nError during inference: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

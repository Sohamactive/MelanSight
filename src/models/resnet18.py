"""
ResNet-18 implementation for MelanSight.

Architecture:
- Pretrained ResNet-18 backbone
- Seven-class classification head
- Optional backbone freezing
- Grad-CAM target layer support
"""

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18
from pathlib import Path


CLASS_NAMES = (
    "akiec",
    "bcc",
    "bkl",
    "df",
    "mel",
    "nv",
    "vasc",
)

NUM_CLASSES = len(CLASS_NAMES)


def create_resnet18(
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> nn.Module:
    """
    Create ResNet-18 for HAM10000 classification.
    """

    if num_classes < 2:
        raise ValueError("num_classes must be at least 2")

    weights = ResNet18_Weights.DEFAULT if pretrained else None

    model = resnet18(weights=weights)

    # Replace ImageNet's 1000-class classifier.
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    # Initialize the new classification layer.
    nn.init.xavier_uniform_(model.fc.weight)
    nn.init.zeros_(model.fc.bias)

    # Optionally freeze the backbone.
    if freeze_backbone:
        for name, parameter in model.named_parameters():
            if not name.startswith("fc."):
                parameter.requires_grad = False

    return model


def get_gradcam_target_layer(model: nn.Module) -> nn.Module:
    """Return the final residual block for Grad-CAM."""

    return model.layer4[-1]


def get_model_info(model: nn.Module) -> dict:
    """Return model architecture and parameter statistics."""

    total_params = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_params = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    return {
        "model_name": "ResNet-18",
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": total_params - trainable_params,
        "input_size": (3, 224, 224),
        "output_size": model.fc.out_features,
    }


def save_model_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_metric: float,
    save_path: str,
) -> None:
    """Save model weights and training state."""

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_metric": val_metric,
        "model_name": "resnet18",
        "num_classes": model.fc.out_features,
        "class_names": CLASS_NAMES,
    }

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(checkpoint, save_path)

    print(f"Checkpoint saved: {save_path}")


def load_model_checkpoint(
    checkpoint_path: str,
    model: nn.Module = None,
    optimizer: torch.optim.Optimizer = None,
    device: str = "cpu",
):
    """Load a previously saved ResNet-18 checkpoint."""

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    if model is None:
        model = create_resnet18(
            num_classes=checkpoint["num_classes"],
            pretrained=False,
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    if optimizer is not None:
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

    return model, checkpoint
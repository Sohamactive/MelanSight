"""VGG-16 model for MelanSight's seven-class HAM10000 task.

The model returns raw logits (not softmax probabilities), as expected by
CrossEntropyLoss and the project's focal-loss implementation.
"""

from __future__ import annotations

from torch import nn
from torchvision.models import VGG16_Weights, vgg16

# This order is the repository-wide contract in AGENTS.md.
CLASS_NAMES: tuple[str, ...] = (
    "akiec",
    "bcc",
    "bkl",
    "df",
    "mel",
    "nv",
    "vasc",
)

NUM_CLASSES = len(CLASS_NAMES)


def create_vgg16(
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
    freeze_features: bool = False,
) -> nn.Module:
    """Build VGG-16 with a classifier sized for the requested classes.

    Args:
        num_classes: Number of output classes. Use 7 for HAM10000.
        pretrained: Initialize from ImageNet V1 weights when True.
            Downloading these weights requires network access the first time.
        freeze_features: If True, freeze convolutional feature extractor
            parameters while leaving the classifier trainable.

    Returns:
        A VGG-16 model whose forward pass returns logits shaped [N, num_classes].
    """
    if num_classes < 2:
        raise ValueError(f"num_classes must be >= 2, got {num_classes}")

    weights = VGG16_Weights.IMAGENET1K_V1 if pretrained else None
    model = vgg16(weights=weights)

    # Keep VGG16's pretrained feature extractor and hidden classifier layers;
    # replace only the final ImageNet 1000-class layer.
    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, num_classes)

    if freeze_features:
        for parameter in model.features.parameters():
            parameter.requires_grad = False

    return model


def get_gradcam_target_layer(model: nn.Module) -> nn.Module:
    """Return VGG-16's final convolutional layer for Grad-CAM."""
    # The last convolution is features[28] in torchvision's VGG-16 layout.
    return model.features[28]


if __name__ == "__main__":
    # Lightweight architecture smoke test; does not download pretrained weights.
    import torch

    model = create_vgg16(pretrained=False)
    model.eval()
    with torch.no_grad():
        output = model(torch.zeros(2, 3, 224, 224))

    assert output.shape == (2, NUM_CLASSES), output.shape
    print(f"VGG16 smoke test passed: output shape = {tuple(output.shape)}")

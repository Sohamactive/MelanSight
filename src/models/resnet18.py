"""
ResNet-18 model implementation for HAM10000 skin lesion classification.

This module provides a ResNet-18 model with a custom 7-class classifier head
for skin lesion classification. Supports ImageNet pretrained weights and
fine-tuning options.

Architecture:
    - ResNet-18 backbone (18 layers, ~11M parameters)
    - Custom FC layer: 512 → 7 classes
    - Input: (B, 3, 224, 224)
    - Output: (B, 7) logits

Example:
    >>> from src.models.resnet18 import create_resnet18
    >>> model = create_resnet18(num_classes=7, pretrained=True)
    >>> print(model)
    >>> 
    >>> # Forward pass
    >>> import torch
    >>> x = torch.randn(2, 3, 224, 224)
    >>> output = model(x)
    >>> print(output.shape)  # (2, 7)

References:
    - ResNet Paper: https://arxiv.org/abs/1512.03385
    - Torchvision: https://pytorch.org/vision/stable/models.html
"""

from pathlib import Path
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
from torchvision import models


def create_resnet18(
    num_classes: int = 7,
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> models.ResNet:
    """
    Create ResNet-18 model with custom classifier for skin lesion classification.
    
    Loads a ResNet-18 architecture with optional ImageNet pretrained weights
    and replaces the final fully connected layer for 7-class classification.
    
    Args:
        num_classes: Number of output classes (default: 7 for HAM10000)
        pretrained: Whether to use ImageNet pretrained weights (default: True)
        freeze_backbone: Whether to freeze backbone layers (default: False)
            If True, only the classifier layer will be trainable.
    
    Returns:
        ResNet-18 model with custom classifier head
    
    Raises:
        ValueError: If num_classes is not positive
    
    Example:
        >>> model = create_resnet18(num_classes=7, pretrained=True)
        >>> 
        >>> # Check model architecture
        >>> print(f"Input: (B, 3, 224, 224)")
        >>> print(f"Output: (B, {num_classes})")
        >>> 
        >>> # Count parameters
        >>> total = sum(p.numel() for p in model.parameters())
        >>> trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        >>> print(f"Total parameters: {total:,}")
        >>> print(f"Trainable parameters: {trainable:,}")
    
    Notes:
        - Input images should be normalized with ImageNet statistics
        - Output are logits (not probabilities), apply softmax for predictions
        - For Grad-CAM, use target_layer='layer4' (see get_gradcam_target_layer())
    """
    if num_classes <= 0:
        raise ValueError(f"num_classes must be positive, got {num_classes}")
    
    # Load pretrained ResNet-18
    # Use ResNet18_Weights.IMAGENET1K_V1 for pretrained weights
    if pretrained:
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)
    else:
        model = models.resnet18(weights=None)
    
    # Get the number of input features for the final layer
    in_features = model.fc.in_features  # Should be 512 for ResNet-18
    
    # Replace the final fully connected layer
    # Original: 512 → 1000 (ImageNet classes)
    # New: 512 → num_classes (HAM10000 classes)
    model.fc = nn.Linear(in_features, num_classes)
    
    # Initialize the new layer with small random weights
    # This helps with training stability
    nn.init.xavier_uniform_(model.fc.weight)
    nn.init.zeros_(model.fc.bias)
    
    # Optionally freeze backbone (feature extractor)
    if freeze_backbone:
        # Freeze all layers except the final classifier
        for name, param in model.named_parameters():
            if 'fc' not in name:  # Don't freeze the classifier layer
                param.requires_grad = False
    
    return model


def get_gradcam_target_layer(model: models.ResNet) -> nn.Module:
    """
    Get the target layer for Grad-CAM visualization.
    
    For ResNet-18, the best layer is 'layer4' which is the final
    convolutional layer before the global average pooling.
    
    Args:
        model: ResNet-18 model instance
    
    Returns:
        Target convolutional layer for Grad-CAM
    
    Example:
        >>> model = create_resnet18()
        >>> target_layer = get_gradcam_target_layer(model)
        >>> print(target_layer)  # layer4[-1]
        >>> 
        >>> # Use with Grad-CAM
        >>> from pytorch_grad_cam import GradCAM
        >>> cam = GradCAM(model=model, target_layers=[target_layer])
    
    Why layer4:
        - layer4 contains the final conv features (512 channels)
        - It's the last layer before global average pooling
        - Provides good spatial resolution for visualization
    """
    # The target layer is the last BasicBlock in layer4
    # This is typically model.layer4[-1] or model.layer4[1] for ResNet-18
    return model.layer4[-1]


def get_model_info(model: models.ResNet) -> dict:
    """
    Get information about the ResNet-18 model.
    
    Returns parameter counts, layer information, and other details.
    
    Args:
        model: ResNet-18 model instance
    
    Returns:
        Dictionary with model information
    
    Example:
        >>> model = create_resnet18()
        >>> info = get_model_info(model)
        >>> print(f"Parameters: {info['total_params']:,}")
        >>> print(f"Layers: {info['num_layers']}")
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Count layers
    num_conv = sum(1 for _ in model.modules() if isinstance(_, nn.Conv2d))
    num_linear = sum(1 for _ in model.modules() if isinstance(_, nn.Linear))
    num_bn = sum(1 for _ in model.modules() if isinstance(_, nn.BatchNorm2d))
    
    return {
        'model_name': 'ResNet-18',
        'total_params': total_params,
        'trainable_params': trainable_params,
        'frozen_params': total_params - trainable_params,
        'num_conv_layers': num_conv,
        'num_linear_layers': num_linear,
        'num_bn_layers': num_bn,
        'input_size': (3, 224, 224),
        'output_size': model.fc.out_features,
    }


def get_layer_groups(model: models.ResNet) -> List[nn.Module]:
    """
    Get layer groups for differential learning rates.
    
    Returns the model split into groups that can use different learning rates:
    - Group 0: Early layers (conv1, bn1, layer1) - lower LR
    - Group 1: Middle layers (layer2, layer3) - medium LR
    - Group 2: Late layers (layer4, fc) - higher LR
    
    Args:
        model: ResNet-18 model instance
    
    Returns:
        List of layer groups
    
    Example:
        >>> model = create_resnet18()
        >>> layer_groups = get_layer_groups(model)
        >>> 
        >>> # Use with optimizer
        >>> optimizer = torch.optim.Adam([
        ...     {'params': layer_groups[0].parameters(), 'lr': 1e-5},
        ...     {'params': layer_groups[1].parameters(), 'lr': 1e-4},
        ...     {'params': layer_groups[2].parameters(), 'lr': 1e-3},
        ... ])
    """
    return [
        # Early layers (lower learning rate)
        nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool, model.layer1),
        # Middle layers (medium learning rate)
        nn.Sequential(model.layer2, model.layer3),
        # Late layers (higher learning rate)
        nn.Sequential(model.layer4, model.fc),
    ]


def save_model_checkpoint(
    model: models.ResNet,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_metric: float,
    save_path: Path,
) -> None:
    """
    Save model checkpoint with training state.
    
    Args:
        model: ResNet-18 model
        optimizer: Optimizer instance
        epoch: Current epoch number
        val_metric: Validation metric (e.g., F1 score)
        save_path: Path to save checkpoint
    
    Example:
        >>> save_model_checkpoint(
        ...     model=model,
        ...     optimizer=optimizer,
        ...     epoch=10,
        ...     val_metric=0.85,
        ...     save_path=Path('checkpoints/resnet18_best.pth')
        ... )
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_metric': val_metric,
        'model_name': 'resnet18',
        'num_classes': model.fc.out_features,
    }
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved: {save_path}")


def load_model_checkpoint(
    checkpoint_path: Path,
    model: Optional[models.ResNet] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = 'cpu',
) -> Tuple[models.ResNet, dict]:
    """
    Load model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Optional model instance (if None, creates new one)
        optimizer: Optional optimizer to load state into
        device: Device to load model onto
    
    Returns:
        Tuple of (model, checkpoint_info)
    
    Example:
        >>> model, info = load_model_checkpoint(
        ...     checkpoint_path=Path('checkpoints/resnet18_best.pth'),
        ...     device='cuda'
        ... )
        >>> print(f"Loaded epoch {info['epoch']} with F1={info['val_metric']:.3f}")
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model if not provided
    if model is None:
        num_classes = checkpoint.get('num_classes', 7)
        model = create_resnet18(num_classes=num_classes, pretrained=False)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return model, checkpoint


# Test when run directly
if __name__ == "__main__":
    """Test ResNet-18 model creation and forward pass."""
    
    print("=" * 60)
    print("Testing ResNet-18 Model")
    print("=" * 60)
    
    # Test 1: Create model
    print("\n1. Creating ResNet-18 with pretrained weights...")
    model = create_resnet18(num_classes=7, pretrained=True)
    print("  ✓ Model created")
    
    # Test 2: Model info
    print("\n2. Model information:")
    info = get_model_info(model)
    print(f"  Total parameters: {info['total_params']:,}")
    print(f"  Trainable parameters: {info['trainable_params']:,}")
    print(f"  Input size: {info['input_size']}")
    print(f"  Output classes: {info['output_size']}")
    
    # Test 3: Forward pass
    print("\n3. Testing forward pass...")
    batch_size = 2
    x = torch.randn(batch_size, 3, 224, 224)
    
    model.eval()
    with torch.no_grad():
        output = model(x)
    
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output type: {output.dtype}")
    assert output.shape == (batch_size, 7), f"Expected (2, 7), got {output.shape}"
    print("  ✓ Forward pass successful")
    
    # Test 4: Grad-CAM target layer
    print("\n4. Grad-CAM target layer:")
    target_layer = get_gradcam_target_layer(model)
    print(f"  Target layer: {target_layer}")
    print("  ✓ Layer found")
    
    # Test 5: Frozen backbone
    print("\n5. Testing frozen backbone...")
    frozen_model = create_resnet18(num_classes=7, pretrained=True, freeze_backbone=True)
    frozen_info = get_model_info(frozen_model)
    print(f"  Total parameters: {frozen_info['total_params']:,}")
    print(f"  Trainable parameters: {frozen_info['trainable_params']:,}")
    print(f"  Frozen parameters: {frozen_info['frozen_params']:,}")
    print("  ✓ Backbone frozen (only classifier trainable)")
    
    # Test 6: Save and load checkpoint
    print("\n6. Testing checkpoint save/load...")
    from torch.optim import Adam
    
    test_checkpoint = Path("checkpoints/test_resnet18.pth")
    optimizer = Adam(model.parameters(), lr=1e-3)
    
    save_model_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=5,
        val_metric=0.85,
        save_path=test_checkpoint,
    )
    
    loaded_model, checkpoint_info = load_model_checkpoint(
        checkpoint_path=test_checkpoint,
        device='cpu',
    )
    
    print(f"  Loaded epoch: {checkpoint_info['epoch']}")
    print(f"  Loaded metric: {checkpoint_info['val_metric']:.3f}")
    print("  ✓ Checkpoint saved and loaded")
    
    # Cleanup
    if test_checkpoint.exists():
        test_checkpoint.unlink()
        print(f"  Cleaned up test checkpoint")
    
    print("\n" + "=" * 60)
    print("✓ All ResNet-18 tests passed!")
    print("=" * 60)
    print("\nModel ready for training!")

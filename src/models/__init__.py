"""
Model architectures for HAM10000 skin lesion classification.

This module provides implementations of three CNN architectures:
- ResNet-18 (ResNet-18): Lightweight, fast training
- VGG-16 (VGG-16): Classic architecture, high capacity
- EfficientNet-B0 (EfficientNet-B0): Efficient, compound scaling

All models:
- Use ImageNet pretrained weights by default
- Have 7-class output heads for HAM10000
- Support Grad-CAM visualization
- Provide consistent interface for training

Quick Start:
    >>> from src.models import create_model
    >>> 
    >>> # Create ResNet-18
    >>> model = create_model('resnet18', num_classes=7, pretrained=True)
    >>> 
    >>> # Forward pass
    >>> import torch
    >>> x = torch.randn(2, 3, 224, 224)
    >>> output = model(x)  # (2, 7)
"""

from pathlib import Path
from typing import Optional, Union, Dict, Any

import torch
import torch.nn as nn

# Import model creation functions
try:
    from src.models.resnet18 import (
        create_resnet18,
        CLASS_NAMES,
        get_gradcam_target_layer as get_resnet_gradcam_layer,
        get_model_info as get_resnet_info,
        save_model_checkpoint,
        load_model_checkpoint,
    )
    RESNET_AVAILABLE = True
except ImportError:
    RESNET_AVAILABLE = False

try:
    from src.models.vgg16 import (
        create_vgg16,
        get_gradcam_target_layer as get_vgg_gradcam_layer,
        get_model_info as get_vgg_info,
    )
    VGG_AVAILABLE = True
except ImportError:
    VGG_AVAILABLE = False

try:
    from src.models.efficientnet_b0 import (
        create_efficientnet_b0,
        get_gradcam_target_layer as get_efficientnet_gradcam_layer,
        get_model_info as get_efficientnet_info,
    )
    EFFICIENTNET_AVAILABLE = True
except ImportError:
    EFFICIENTNET_AVAILABLE = False


# Model registry
MODEL_REGISTRY = {}

if RESNET_AVAILABLE:
    MODEL_REGISTRY['resnet18'] = {
        'creator': create_resnet18,
        'gradcam_layer': get_resnet_gradcam_layer,
        'info': get_resnet_info,
        'description': 'ResNet-18: Lightweight, fast training',
    }

if VGG_AVAILABLE:
    MODEL_REGISTRY['vgg16'] = {
        'creator': create_vgg16,
        'gradcam_layer': get_vgg_gradcam_layer,
        'info': get_vgg_info,
        'description': 'VGG-16: Classic architecture, high capacity',
    }

if EFFICIENTNET_AVAILABLE:
    MODEL_REGISTRY['efficientnet_b0'] = {
        'creator': create_efficientnet_b0,
        'gradcam_layer': get_efficientnet_gradcam_layer,
        'info': get_efficientnet_info,
        'description': 'EfficientNet-B0: Efficient compound scaling',
    }


def create_model(
    model_name: str,
    num_classes: int = 7,
    pretrained: bool = True,
    **kwargs,
) -> nn.Module:
    """
    Create a model by name.
    
    Unified interface for creating any supported model architecture.
    
    Args:
        model_name: One of 'resnet18', 'vgg16', 'efficientnet_b0'
        num_classes: Number of output classes (default: 7)
        pretrained: Whether to use ImageNet pretrained weights (default: True)
        **kwargs: Additional arguments passed to model creator
    
    Returns:
        Created model instance
    
    Raises:
        ValueError: If model_name is not recognized
    
    Example:
        >>> model = create_model('resnet18', num_classes=7, pretrained=True)
        >>> print(type(model))
        >>> 
        >>> # With frozen backbone
        >>> model = create_model('resnet18', freeze_backbone=True)
        >>> 
        >>> # All models have same interface
        >>> for name in ['resnet18', 'vgg16', 'efficientnet_b0']:
        ...     model = create_model(name, num_classes=7)
    """
    model_name = model_name.lower()
    
    if model_name not in MODEL_REGISTRY:
        available = ', '.join(MODEL_REGISTRY.keys())
        raise ValueError(
            f"Unknown model: '{model_name}'. "
            f"Available: {available}"
        )
    
    creator = MODEL_REGISTRY[model_name]['creator']
    return creator(num_classes=num_classes, pretrained=pretrained, **kwargs)


def get_available_models() -> Dict[str, str]:
    """
    Get list of available models with descriptions.
    
    Returns:
        Dictionary mapping model names to descriptions
    
    Example:
        >>> models = get_available_models()
        >>> for name, desc in models.items():
        ...     print(f"{name}: {desc}")
    """
    return {
        name: info['description']
        for name, info in MODEL_REGISTRY.items()
    }


def get_model_summary(model: nn.Module) -> str:
    """
    Get a string summary of the model.
    
    Args:
        model: PyTorch model
    
    Returns:
        Formatted summary string
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    lines = [
        "=" * 60,
        "Model Summary",
        "=" * 60,
        f"Architecture: {model.__class__.__name__}",
        f"Total parameters: {total_params:,}",
        f"Trainable parameters: {trainable_params:,}",
        f"Frozen parameters: {total_params - trainable_params:,}",
        "=" * 60,
    ]
    
    return '\n'.join(lines)


# Exports
__all__ = [
    'CLASS_NAMES',
    # Model creation
    'create_model',
    'get_available_models',
    'get_model_summary',
    
    # ResNet-18 (if available)
    'create_resnet18',
    'get_resnet_gradcam_layer',
    'get_resnet_info',
    
    # Checkpoint utilities
    'save_model_checkpoint',
    'load_model_checkpoint',
]

# Conditionally add to __all__
if RESNET_AVAILABLE:
    __all__.extend(['create_resnet18', 'get_resnet_gradcam_layer', 'get_resnet_info'])
if VGG_AVAILABLE:
    __all__.extend(['create_vgg16', 'get_vgg_gradcam_layer', 'get_vgg_info'])
if EFFICIENTNET_AVAILABLE:
    __all__.extend(['create_efficientnet_b0', 'get_efficientnet_gradcam_layer', 'get_efficientnet_info'])


# Test when run directly
if __name__ == "__main__":
    """Test model module."""
    
    print("=" * 60)
    print("Testing Models Module")
    print("=" * 60)
    
    print("\nAvailable models:")
    for name, desc in get_available_models().items():
        print(f"  ✓ {name}: {desc}")
    
    print(f"\nTotal available: {len(MODEL_REGISTRY)}")
    
    if len(MODEL_REGISTRY) == 0:
        print("\n  No models available yet.")
        print("  Install PyTorch and torchvision:")
        print("    uv add torch torchvision")
        print("  Then create model files:")
        print("    - src/models/resnet18.py")
        print("    - src/models/vgg16.py")
        print("    - src/models/efficientnet_b0.py")
    
    # Test creating a model if available
    if 'resnet18' in MODEL_REGISTRY:
        print("\n" + "=" * 60)
        print("Testing ResNet-18 Creation")
        print("=" * 60)
        
        model = create_model('resnet18', num_classes=7, pretrained=False)
        print(get_model_summary(model))
        
        # Test forward pass
        import torch
        x = torch.randn(1, 3, 224, 224)
        y = model(x)
        print(f"\nForward pass: {x.shape} → {y.shape}")
        print("✓ Model working!")
    
    print("\n" + "=" * 60)
    print("✓ Models module ready!")
    print("=" * 60)
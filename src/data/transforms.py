"""
Image transforms for HAM10000 skin lesion classification.

This module defines preprocessing and augmentation pipelines for training,
validation, and test sets. The transforms are designed specifically for
dermoscopic images and pretrained model compatibility.

Training transforms include data augmentation to improve generalization.
Validation/test transforms are deterministic for consistent evaluation.

Example:
    >>> from src.data.transforms import get_train_transforms, get_val_transforms
    >>> train_transform = get_train_transforms()
    >>> val_transform = get_val_transforms()
    >>> 
    >>> # Apply to PIL Image
    >>> image_tensor = train_transform(pil_image)  # Shape: (3, 224, 224)
"""

# from typing import tuple

from pathlib import Path

import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from torchvision import transforms


#shouldn't the ↓ be in __init__.py ???


# ImageNet normalization statistics
# Pretrained models expect inputs normalized with these values
IMAGENET_MEAN = (0.485, 0.456, 0.406)  # RGB channel means
IMAGENET_STD = (0.229, 0.224, 0.225)   # RGB channel stds

# Image size for pretrained models
IMAGE_SIZE = 224  # Standard for ResNet, VGG, EfficientNet


def get_train_transforms(
    image_size: int = IMAGE_SIZE,
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
) -> transforms.Compose:
    """
    Get training transforms with data augmentation.
    
    Training transforms include random augmentations to increase dataset
    diversity and improve model generalization. Each training sample
    will be augmented differently each epoch.
    
    Augmentations applied (in order):
    1. Resize to target size (maintains aspect ratio, then crops)
    2. Random horizontal flip (50% probability)
    3. Random rotation (±15 degrees)
    4. Random brightness/contrast adjustment
    5. Convert to tensor (0-255 → 0-1)
    6. Normalize with ImageNet statistics
    
    Args:
        image_size: Target image size (default: 224)
        mean: Normalization mean for eac channel
        std: Normalization std for each channel
    
    Returns:
        Composed transforms for training
    
    Example:
        >>> transform = get_train_transforms()
        >>> image_tensor = transform(pil_image)  # (3, 224, 224)
        >>> print(image_tensor.min(), image_tensor.max())  # ~-2.0 to ~2.0
    
    Why these augmentations:
    - Horizontal flip: Lesions can appear on either side of body
    - Rotation: Camera orientation varies in dermoscopic images
    - Color jitter: Lighting conditions vary between images
    - Resize: Pretrained models expect fixed input size
    """
    return transforms.Compose([
        # Resize to target size
        # Uses bilinear interpolation by default
        transforms.Resize((image_size, image_size)),
        
        # Random horizontal flip
        # Probability: 50%
        # Why: Lesions can appear on left or right side
        transforms.RandomHorizontalFlip(p=0.5),
        
        # Random rotation
        # Range: ±15 degrees
        # Why: Camera orientation varies in dermoscopic imaging
        transforms.RandomRotation(degrees=15),
        
        # Color jitter (brightness and contrast)
        # Brightness: ±10% (0.9 to 1.1)
        # Contrast: ±10% (0.9 to 1.1)
        # Why: Lighting conditions vary between images
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        
        # Convert PIL Image to tensor
        # Scales pixel values from [0, 255] to [0.0, 1.0]
        transforms.ToTensor(),
        
        # Normalize with ImageNet statistics
        # Formula: (x - mean) / std
        # Results in approximately zero mean, unit variance
        # Required for pretrained model compatibility
        transforms.Normalize(mean=mean, std=std),
    ])


def get_val_transforms(
    image_size: int = IMAGE_SIZE,
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
) -> transforms.Compose:
    """
    Get validation/test transforms (deterministic, no augmentation).
    
    Validation and test transforms must be deterministic to ensure
    consistent and reproducible evaluation. No random operations.
    
    Transforms applied:
    1. Resize to target size
    2. Convert to tensor
    3. Normalize with ImageNet statistics
    
    Args:
        image_size: Target image size (default: 224)
        mean: Normalization mean for each channel
        std: Normalization std for each channel
    
    Returns:
        Composed transforms for validation/test
    
    Example:
        >>> transform = get_val_transforms()
        >>> image_tensor = transform(pil_image)  # (3, 224, 224)
    
    Important:
        - No random operations (consistent across runs)
        - Same normalization as training (required!)
        - Used for BOTH validation and test sets
    """
    return transforms.Compose([
        # Resize to target size
        transforms.Resize((image_size, image_size)),
        
        # Convert to tensor
        transforms.ToTensor(),
        
        # Normalize with ImageNet statistics
        # Must match training normalization!
        transforms.Normalize(mean=mean, std=std),
    ])


def get_test_transforms(
    image_size: int = IMAGE_SIZE,
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
) -> transforms.Compose:
    """
    Get test transforms (same as validation).
    
    Test transforms are identical to validation transforms.
    Both use deterministic preprocessing without augmentation.
    
    Args:
        image_size: Target image size (default: 224)
        mean: Normalization mean for each channel
        std: Normalization std for each channel
    
    Returns:
        Composed transforms for test
    
    Note:
        This is an alias for get_val_transforms() for clarity.
    """
    return get_val_transforms(image_size=image_size, mean=mean, std=std)


def denormalize(
    tensor: torch.Tensor,
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
) -> torch.Tensor:
    """
    Denormalize a normalized tensor for visualization.
    
    Reverses normalization: x = x * std + mean
    Results in tensor with values in [0, 1] range suitable for display.
    
    Args:
        tensor: Normalized tensor of shape (C, H, W) or (B, C, H, W)
        mean: Original mean used for normalization
        std: Original std used for normalization
    
    Returns:
        Denormalized tensor with values in [0, 1]
    """
    # ✅ Fixed Pylance error by using separate variable names
    # ✅ Fixed GPU issue by matching the input tensor's device
    mean_t = torch.tensor(mean, dtype=torch.float32, device=tensor.device).view(-1, 1, 1)
    std_t = torch.tensor(std, dtype=torch.float32, device=tensor.device).view(-1, 1, 1)
    
    # Clone to avoid modifying original
    denorm = tensor.clone()
    
    # Denormalize
    denorm = denorm * std_t + mean_t
    
    # Clamp to valid range [0, 1]
    denorm = torch.clamp(denorm, 0, 1)
    
    return denorm


# Simple test when run directly
if __name__ == "__main__":
    """Test transforms by applying to a sample image."""
    
    print("=" * 60)
    print("Testing Transforms")
    print("=" * 60)
    
    train_csv = Path("data/splits/train.csv")
    if train_csv.exists():
        # ✅ Fixed blind exception: Only catch errors expected during file loading/parsing
        try:
            df = pd.read_csv(train_csv)
        except (pd.errors.ParserError, pd.errors.EmptyDataError, OSError) as e:
            print(f"\n✗ Failed to load train CSV: {e}")
            df = None

        if df is not None:
            sample_path = Path(df['image_path'].iloc[0])
            
            # Make relative to project root if needed
            if not sample_path.exists():
                sample_path = Path(".") / sample_path
            
            if sample_path.exists():
                print(f"\nLoading sample image: {sample_path}")
                
                # ✅ Fixed blind exception for PIL image loading
                try:
                    image = Image.open(sample_path).convert('RGB')
                except (UnidentifiedImageError, OSError) as e:
                    print(f"\n✗ Failed to open image: {e}")
                    image = None
                
                if image is not None:
                    print(f"Original size: {image.size}")
                    
                    # Test training transforms
                    print("\n" + "-" * 60)
                    print("Training Transforms:")
                    train_transform = get_train_transforms()
                    
                    # Apply twice to show randomness
                    train_img1 = train_transform(image)
                    train_img2 = train_transform(image)
                    
                    print(f"  Output shape: {train_img1.shape}")
                    print(f"  Value range: [{train_img1.min():.3f}, {train_img1.max():.3f}]")
                    print(f"  Random augmentations working: {not torch.allclose(train_img1, train_img2)}")
                    
                    # Test validation transforms
                    print("\n" + "-" * 60)
                    print("Validation Transforms:")
                    val_transform = get_val_transforms()
                    
                    val_img1 = val_transform(image)
                    val_img2 = val_transform(image)
                    
                    print(f"  Output shape: {val_img1.shape}")
                    print(f"  Value range: [{val_img1.min():.3f}, {val_img1.max():.3f}]")
                    print(f"  Deterministic: {torch.allclose(val_img1, val_img2)}")
                    
                    # Test denormalize
                    print("\n" + "-" * 60)
                    print("Denormalize:")
                    denorm_img = denormalize(val_img1)
                    print(f"  After denormalize: [{denorm_img.min():.3f}, {denorm_img.max():.3f}]")
                    print(f"  Ready for visualization: {0 <= denorm_img.min() and denorm_img.max() <= 1}")
                    
                    print("\n" + "=" * 60)
                    print("✓ All transform tests passed!")
                    print("=" * 60)
            else:
                print(f"Sample image not found: {sample_path}")
                print("Run notebooks/02_data_validation.ipynb and 03_data_splitting.ipynb first")
    else:
        print(f"Train CSV not found: {train_csv}")
        print("Run data splitting first")


"""
Data loading module for HAM10000 skin lesion classification.

This module provides a complete data loading pipeline:
- Dataset: Load images and labels from CSV files
- Transforms: Preprocessing and augmentation
- DataLoader: Batch creation with parallel loading

Quick Start:
    >>> from src.data import get_dataloaders
    >>> train_loader, val_loader, test_loader = get_dataloaders(batch_size=16)
    >>> 
    >>> for images, labels in train_loader:
    ...     print(images.shape)  # (16, 3, 224, 224)
    ...     break

Components:
    - SkinLesionDataset: Custom Dataset for HAM10000
    - get_train_transforms: Training augmentation pipeline
    - get_val_transforms: Validation/test preprocessing (no augmentation)
    - get_dataloaders: Create train/val/test DataLoaders

Example Training Setup:
    >>> from src.data import get_dataloaders
    >>> 
    >>> # Create DataLoaders
    >>> train_loader, val_loader, test_loader = get_dataloaders(
    ...     batch_size=16,
    ...     num_workers=4,  # Use 0 on Windows
    ... )
    >>> 
    >>> # Training loop
    >>> for epoch in range(10):
    ...     for images, labels in train_loader:
    ...         # Training step
    ...         pass
"""

from pathlib import Path
from typing import Tuple

# Import main components for easy access
from src.data.dataset import SkinLesionDataset, get_dataset
from src.data.transforms import (
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    denormalize,
    get_test_transforms,
    get_train_transforms,
    get_val_transforms,
)
from src.data.dataloader import get_dataloaders, get_single_loader

__version__ = "0.1.0"

__all__ = [
    # Constants
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "IMAGE_SIZE",
    # Dataset
    "SkinLesionDataset",
    "get_dataset",
    # Transforms
    "denormalize",
    "get_train_transforms",
    "get_val_transforms",
    "get_test_transforms",
    # DataLoader
    "get_dataloaders",
    "get_single_loader",
    "create_data_loaders",
]


def create_data_loaders(
    data_dir: Path = Path("."),
    batch_size: int = 16,
    num_workers: int = 4,
) -> Tuple:
    """
    Convenience function to create all data loaders.
    
    This is a shortcut for get_dataloaders() with better error handling.
    
    Args:
        data_dir: Project root directory
        batch_size: Samples per batch
        num_workers: Parallel loading workers (use 0 on Windows)
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    
    Raises:
        RuntimeError: If data splits don't exist
    
    Example:
        >>> from src.data import create_data_loaders
        >>> train_loader, val_loader, test_loader = create_data_loaders()
    """
    # Check if splits exist
    splits_dir = data_dir / "data" / "splits"
    required_files = ["train.csv", "val.csv", "test.csv"]

    for filename in required_files:
        if not (splits_dir / filename).exists():
            raise RuntimeError(
                f"Missing split file: {splits_dir / filename}\n"
                "Run notebooks/03_data_splitting.ipynb first"
            )

    return get_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
    )


# Simple test
if __name__ == "__main__":
    """Test data module imports."""
    print("=" * 60)
    print("Testing Data Module Imports")
    print("=" * 60)
    
    print("\nImport mode: package (src.data)")
    
    print("\nAvailable components:")
    available_count = 0
    for name in __all__:
        if name in globals():
            print(f"  ✓ {name}")
            available_count += 1
        else:
            print(f"  ✗ {name} (not available)")
    
    print(f"\nTotal available: {available_count}/{len(__all__)}")
    
    print("\n" + "=" * 60)
    if available_count == len(__all__):
        print("✓ Data module ready!")
    print("=" * 60)

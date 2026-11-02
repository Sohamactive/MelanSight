"""
DataLoader creation for HAM10000 skin lesion classification.

This module creates PyTorch DataLoaders for train, validation, and test sets.
DataLoaders handle batching, shuffling, and parallel data loading.

Example:
    >>> from src.data.dataloader import get_dataloaders
    >>> train_loader, val_loader, test_loader = get_dataloaders(batch_size=16)
    >>> 
    >>> for images, labels in train_loader:
    ...     print(images.shape)  # (16, 3, 224, 224)
    ...     break
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Handle both package imports and direct execution
try:
    from src.data.dataset import SkinLesionDataset
    from src.data.transforms import get_train_transforms, get_val_transforms
except ModuleNotFoundError:
    # Running directly - use relative imports
    from dataset import SkinLesionDataset
    from transforms import get_train_transforms, get_val_transforms


def get_dataloaders(
    data_dir: Path = Path("."),
    batch_size: int = 16,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders for train, validation, and test sets.
    
    Creates three DataLoaders with appropriate settings for each split:
    - Train: shuffled, augmented
    - Val: sequential, no augmentation
    - Test: sequential, no augmentation
    
    Args:
        data_dir: Project root directory (default: current directory)
        batch_size: Number of samples per batch (default: 16)
        num_workers: Parallel loading workers (default: 4, use 0 on Windows)
        pin_memory: Pin memory for faster GPU transfer (default: True)
    
    Returns:
        tuple of (train_loader, val_loader, test_loader)
    
    Example:
        >>> train_loader, val_loader, test_loader = get_dataloaders(batch_size=32)
        >>> print(len(train_loader))  # Number of batches
        >>> for images, labels in train_loader:
        ...     # Training loop
        ...     pass
    
    Note:
        On Windows, use num_workers=0 to avoid multiprocessing issues.
        On Linux/Mac, num_workers=4 is recommended for faster loading.
    """
    
    # Paths to split CSVs
    splits_dir = data_dir / "data" / "splits"
    
    # Create datasets with appropriate transforms
    train_dataset = SkinLesionDataset(
        csv_path=splits_dir / "train.csv",
        data_dir=data_dir,
        transform=get_train_transforms(),
    )
    
    val_dataset = SkinLesionDataset(
        csv_path=splits_dir / "val.csv",
        data_dir=data_dir,
        transform=get_val_transforms(),
    )
    
    test_dataset = SkinLesionDataset(
        csv_path=splits_dir / "test.csv",
        data_dir=data_dir,
        transform=get_val_transforms(),  # Same as validation
    )
    
    # Create DataLoaders
    
    # Training: shuffle=True (randomize order each epoch)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,           # Randomize for training
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,        # Keep incomplete batch
    )
    
    # Validation: shuffle=False (consistent evaluation)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,          # Fixed order for evaluation
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    
    # Test: shuffle=False (consistent evaluation)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,          # Fixed order for evaluation
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    
    print("\nDataLoaders created:")
    print(f"  Train:   {len(train_loader)} batches × {batch_size} = ~{len(train_dataset)} samples")
    print(f"  Val:     {len(val_loader)} batches × {batch_size} = ~{len(val_dataset)} samples")
    print(f"  Test:    {len(test_loader)} batches × {batch_size} = ~{len(test_dataset)} samples")
    
    return train_loader, val_loader, test_loader


def get_single_loader(
    csv_path: Path,
    data_dir: Path = Path("."),
    batch_size: int = 16,
    shuffle: bool = False,
    num_workers: int = 4,
    transform=None,
) -> DataLoader:
    """
    Create a single DataLoader for a custom CSV.
    
    Useful for inference on new data or custom splits.
    
    Args:
        csv_path: Path to CSV file with image metadata
        data_dir: Project root directory
        batch_size: Samples per batch
        shuffle: Whether to shuffle data
        num_workers: Parallel loading workers
        transform: Transforms to apply (default: validation transforms)
    
    Returns:
        Configured DataLoader
    
    Example:
        >>> loader = get_single_loader(
        ...     csv_path='data/splits/test.csv',
        ...     batch_size=32
        ... )
    """
    # Import here to avoid circular imports and handle direct execution
    try:
        from src.data.transforms import get_val_transforms
    except ModuleNotFoundError:
        from transforms import get_val_transforms
    
    if transform is None:
        transform = get_val_transforms()
    
    dataset = SkinLesionDataset(
        csv_path=csv_path,
        data_dir=data_dir,
        transform=transform,
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


# # Test when run directly
# if __name__ == "__main__":
#     """Test DataLoaders by creating them and loading one batch."""
    
#     print("=" * 60)
#     print("Testing DataLoaders")
#     print("=" * 60)
    
#     try:
#         # Create DataLoaders
#         # Use num_workers=0 on Windows, 4 on Linux/Mac
#         import sys
#         num_workers = 0 if sys.platform == "win32" else 4
        
#         train_loader, val_loader, test_loader = get_dataloaders(
#             batch_size=16,
#             num_workers=num_workers,
#         )
        
#         print("\n" + "-" * 60)
#         print("Testing Train Loader:")
#         print("-" * 60)
        
#         # Load one batch
#         train_images, train_labels = next(iter(train_loader))
        
#         print(f"  Batch shape: {train_images.shape}")
#         print(f"  Labels shape: {train_labels.shape}")
#         print(f"  Image dtype: {train_images.dtype}")
#         print(f"  Label dtype: {train_labels.dtype}")
#         print(f"  Value range: [{train_images.min():.3f}, {train_images.max():.3f}]")
        
#         print("\n" + "-" * 60)
#         print("Testing Val Loader:")
#         print("-" * 60)
        
#         val_images, val_labels = next(iter(val_loader))
        
#         print(f"  Batch shape: {val_images.shape}")
#         print(f"  Labels shape: {val_labels.shape}")
        
#         print("\n" + "-" * 60)
#         print("Testing Test Loader:")
#         print("-" * 60)
        
#         test_images, test_labels = next(iter(test_loader))
        
#         print(f"  Batch shape: {test_images.shape}")
#         print(f"  Labels shape: {test_labels.shape}")
        
#         print("\n" + "=" * 60)
#         print("✓ All DataLoader tests passed!")
#         print("=" * 60)
#         print("\nReady for training!")
        
#     except Exception as e:  # noqa: BLE001
#         print(f"\n✗ Test failed: {e}")
#         print("\nMake sure to:")
#         print("  1. Run notebooks/02_data_validation.ipynb")
#         print("  2. Run notebooks/03_data_splitting.ipynb")
#         print("  3. Check that CSV files exist in data/splits/")
if __name__ == "__main__":
    import time
    import sys

    print("=" * 60)
    print("DataLoader Worker Benchmark")
    print("=" * 60)

    # Use the training split for benchmarking
    csv_path = Path("data/splits/train.csv")

    # Windows: keep multiprocessing protected by this main block
    worker_options = [0, 4]

    batch_size = 16

    results = {}

    for workers in worker_options:
        print(f"\nTesting num_workers={workers}")

        loader = get_single_loader(
            csv_path=csv_path,
            data_dir=Path("."),
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
        )

        # Warm-up: load one batch before measuring
        iterator = iter(loader)
        try:
            next(iterator)
        except StopIteration:
            raise RuntimeError("Dataset is empty.")

        # Measure complete dataset iteration
        start_time = time.perf_counter()

        total_images = 0
        total_batches = 0

        for images, labels in loader:
            total_images += images.size(0)
            total_batches += 1

        elapsed = time.perf_counter() - start_time

        results[workers] = elapsed

        print(f"  Images loaded: {total_images}")
        print(f"  Batches loaded: {total_batches}")
        print(f"  Time taken: {elapsed:.2f} seconds")

    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)

    for workers, elapsed in results.items():
        print(f"num_workers={workers}: {elapsed:.2f} seconds")

    if results[0] > results[4]:
        speedup = results[0] / results[4]
        print(f"\nnum_workers=4 was {speedup:.2f}x faster.")
    elif results[4] > results[0]:
        slowdown = results[4] / results[0]
        print(f"\nnum_workers=4 was {slowdown:.2f}x slower.")
    else:
        print("\nBoth configurations took approximately the same time.")

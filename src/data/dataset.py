"""
HAM10000 Skin Lesion Dataset

This module defines a custom PyTorch Dataset for loading skin lesion images
and their corresponding labels from CSV split manifests.

The dataset loads images on-demand (not all at once) to conserve memory,
making it suitable for large datasets that don't fit in RAM.

Example:
    >>> from src.data.dataset import SkinLesionDataset
    >>> dataset = SkinLesionDataset('data/splits/train.csv')
    >>> image, label = dataset[0]
    >>> print(image.shape)  # (3, 224, 224)
    >>> print(label)        # 0 (akiec)
"""

from collections.abc import Callable
from pathlib import Path
from typing import ClassVar

import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset


class SkinLesionDataset(Dataset):
    """
    PyTorch Dataset for HAM10000 skin lesion classification.

    Loads images and labels from CSV files created during data splitting.
    Supports custom transforms for data augmentation and preprocessing.

    Args:
        csv_path: Path to CSV file containing image paths and labels
            Expected columns: 'image_path', 'dx', optionally 'lesion_id', 'image_id'
        transform: Optional transform to apply to images
        target_transform: Optional transform to apply to labels

    Attributes:
        num_classes: Number of classes (7 for HAM10000)
        class_names: List of class names in order
        class_to_idx: Mapping from class name to integer index
        idx_to_class: Mapping from integer index to class name

    Example:
        >>> dataset = SkinLesionDataset(
        ...     csv_path='data/splits/train.csv',
        ...     transform=train_transforms
        ... )
        >>> print(len(dataset))  # 7010
        >>> image, label = dataset[0]
        >>> print(image.shape)   # torch.Size([3, 224, 224])
    """

    # HAM10000 class names and their full descriptions
    CLASS_NAMES: ClassVar[list[str]] = [
        "akiec",  # Actinic keratoses and intraepithelial carcinoma
        "bcc",  # Basal cell carcinoma
        "bkl",  # Benign keratosis-like lesions
        "df",  # Dermatofibroma
        "mel",  # Melanoma
        "nv",  # Melanocytic nevi
        "vasc",  # Vascular lesions
    ]

    # Class descriptions for reference
    CLASS_DESCRIPTIONS: ClassVar[dict[str, str]] = {
        "akiec": "Actinic keratoses and intraepithelial carcinoma",
        "bcc": "Basal cell carcinoma",
        "bkl": "Benign keratosis-like lesions",
        "df": "Dermatofibroma",
        "mel": "Melanoma",
        "nv": "Melanocytic nevi",
        "vasc": "Vascular lesions",
    }

    def __init__(
        self,
        csv_path: str | Path,
        data_dir: str | Path = ".",
        transform: Callable | None = None,
        target_transform: Callable | None = None,
    ):
        """
        Initialize the dataset.

        Args:
            csv_path: Path to CSV file with image metadata
            data_dir: Root directory for resolving relative image paths (default: current directory)
            transform: Optional transform callable for images
            target_transform: Optional transform callable for labels

        Raises:
            FileNotFoundError: If CSV file doesn't exist
            ValueError: If required columns are missing
        """
        super().__init__()

        # Convert to Path object for better handling
        self.csv_path = Path(csv_path)
        self.data_dir = Path(data_dir).resolve()

        # Validate file exists
        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"CSV file not found: {self.csv_path}\n"
                f"Run data splitting first: notebooks/03_data_splitting.ipynb"
            )

        # Load the CSV
        try:
            self.df = pd.read_csv(self.csv_path)
        except (pd.errors.ParserError, pd.errors.EmptyDataError, OSError) as e:
            raise ValueError(f"Failed to load or parse CSV {self.csv_path}: {e}") from e


        # Validate required columns
        required_cols = ["image_path", "dx"]
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(
                f"CSV missing required columns: {missing_cols}\n"
                f"Found columns: {list(self.df.columns)}"
            )

        # Store transforms
        self.transform = transform
        self.target_transform = target_transform    

        # Create class mappings (consistent across all splits)
        self.num_classes = len(self.CLASS_NAMES)
        self.class_to_idx = {
            cls_name: idx for idx, cls_name in enumerate(self.CLASS_NAMES)
        }
        self.idx_to_class = {
            idx: cls_name for idx, cls_name in enumerate(self.CLASS_NAMES)
        }

        # Validate all classes in data are valid
        unique_classes = set(self.df["dx"].unique())
        expected_classes = set(self.CLASS_NAMES)

        if not unique_classes.issubset(expected_classes):
            unexpected = unique_classes - expected_classes
            raise ValueError(
                f"CSV contains unexpected classes: {unexpected}\n"
                f"Expected classes: {expected_classes}"
            )

        # Print dataset info
        self._print_dataset_info()

    def _print_dataset_info(self) -> None:
        """Print summary information about the dataset."""
        print(f"\nDataset loaded: {self.csv_path.name}")
        print(f"  Total samples: {len(self)}")
        print(f"  Number of classes: {self.num_classes}")
        print(f"  Classes: {self.CLASS_NAMES}")

        # Print class distribution
        class_counts = self.df["dx"].value_counts().sort_index()
        print("\n  Class distribution:")
        for cls_name in self.CLASS_NAMES:
            count = class_counts.get(cls_name, 0)
            pct = count / len(self) * 100
            desc = self.CLASS_DESCRIPTIONS[cls_name][:30]
            print(f"    {cls_name:6s}: {count:5d} ({pct:5.1f}%) - {desc}")

    def __len__(self) -> int:
        """
        Return the total number of samples in the dataset.

        Returns:
            Number of images in the dataset
        """
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """
        Get a single sample (image and label) by index.

        This method is called by DataLoader to load individual samples.
        It loads the image from disk on-demand (not cached).

        Args:
            idx: Index of the sample to retrieve

        Returns:
            tuple of (image_tensor, label)
            - image_tensor: Tensor of shape (C, H, W), values normalized
            - label: Integer class index (0-6)

        Raises:
            IndexError: If idx is out of bounds
            FileNotFoundError: If image file doesn't exist
        """
        # Validate index
        if not 0 <= idx < len(self):
            raise IndexError(f"Index {idx} out of range [0, {len(self)})")

        # Get row from dataframe
        row = self.df.iloc[idx]

        # Get image path and label
        # Resolve relative paths against data_dir for cross-machine compatibility
        image_path = Path(row["image_path"])
        if not image_path.is_absolute():
            image_path = self.data_dir / image_path
        class_name = row["dx"]

        # Validate image exists
        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}\n"
                f"Check data split CSV: {self.csv_path}"
            )

        # Load image
        try:
            # Open image and convert to RGB (handles grayscale or RGBA)
            image = Image.open(image_path).convert("RGB")
        except (UnidentifiedImageError, OSError) as e:
            raise RuntimeError(f"Failed to load image {image_path}: {e}")

        # Get label as integer index
        label = self.class_to_idx[class_name]

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label # type: ignore

    def get_class_name(self, idx: int) -> str:
        """
        Get class name from integer index.

        Args:
            idx: Integer class index (0-6)

        Returns:
            Class name string (e.g., 'akiec', 'mel')

        Raises:
            KeyError: If idx is not a valid class index
        """
        if idx not in self.idx_to_class:
            raise KeyError(
                f"Invalid class index: {idx}. Valid range: 0-{self.num_classes-1}"
            )
        return self.idx_to_class[idx]

    def get_class_description(self, idx_or_name: int | str) -> str:
        """
        Get full description of a class.

        Args:
            idx_or_name: Class index (int) or class name (str)

        Returns:
            Full description of the class
        """
        if isinstance(idx_or_name, int):
            class_name = self.get_class_name(idx_or_name)
        else:
            class_name = idx_or_name

        return self.CLASS_DESCRIPTIONS.get(class_name, "Unknown class")

    def get_sample_info(self, idx: int) -> dict:
        """
        Get detailed information about a sample.

        Useful for debugging and analysis.

        Args:
            idx: Sample index

        Returns:
            Dictionary with sample metadata
        """
        row = self.df.iloc[idx]

        return {
            "index": idx,
            "lesion_id": row.get("lesion_id", "N/A"),
            "image_id": row.get("image_id", "N/A"),
            "image_path": row["image_path"],
            "class_name": row["dx"],
            "class_idx": self.class_to_idx[row["dx"]],
            "class_description": self.CLASS_DESCRIPTIONS[row["dx"]],
            "age": row.get("age", "N/A"),
            "sex": row.get("sex", "N/A"),
            "localization": row.get("localization", "N/A"),
        }


def get_dataset(
    split: str,
    data_dir: str | Path = "data/splits",
    root_dir: str | Path = ".",
    transform: Callable | None = None,
) -> SkinLesionDataset:
    """
    Convenience function to load a dataset split.

    Args:
        split: Which split to load ('train', 'val', or 'test')
        data_dir: Directory containing split CSV files
        root_dir: Root directory for resolving relative image paths (default: current directory)
        transform: Optional transform to apply

    Returns:
        SkinLesionDataset instance

    Example:
        >>> train_dataset = get_dataset('train', transform=train_transforms)
        >>> val_dataset = get_dataset('val', transform=val_transforms)
    """
    data_dir = Path(data_dir)
    csv_path = data_dir / f"{split}.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Split file not found: {csv_path}\n"
            f"Available splits: train.csv, val.csv, test.csv"
        )

    return SkinLesionDataset(csv_path, data_dir=root_dir, transform=transform)


# Testing code (run directly to test)
if __name__ == "__main__":
    """
    Test the dataset by loading a sample.

    Run with:
        uv run python src/data/dataset.py
    """
    print("=" * 60)
    print("Testing SkinLesionDataset")
    print("=" * 60)

    # Test loading train split
    try:
        dataset = SkinLesionDataset("data/splits/train.csv")
        print("\n✓ Dataset loaded successfully!")
        print(f"  Length: {len(dataset)}")

        # Test getting a sample
        image, label = dataset[0]
        print("\n✓ Sample loaded:")
        print(f"  Image type: {type(image)}")
        if hasattr(image, "shape"):
            print(f"  Image shape: {image.shape}")
        print(f"  Label: {label} ({dataset.get_class_name(label)})")

        # Test get_sample_info
        info = dataset.get_sample_info(0)
        print("\n✓ Sample info:")
        for key, value in info.items():
            print(f"  {key}: {value}")

        # Test class mappings
        print("\n✓ Class mappings:")
        for idx, name in dataset.idx_to_class.items():
            desc = dataset.CLASS_DESCRIPTIONS[name]
            print(f"  {idx}: {name} - {desc}")

        print("\n" + "=" * 60)
        print("All tests passed!")
        print("=" * 60)

    except FileNotFoundError as e:
        print("\n✗ Test failed:")
        print(f"  {e}")
        print("\n  Make sure to run data splitting first:")
        print("    notebooks/03_data_splitting.ipynb")
    except Exception as e:  # noqa: BLE001
        print("\n✗ Test failed:")
        print(f"  {type(e).__name__}: {e}")
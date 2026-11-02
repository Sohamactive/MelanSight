"""Read-only audit of the HAM10000 split manifests; exits non-zero on any failed check."""

import json
import sys
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

from src.data.transforms import IMAGENET_MEAN, IMAGENET_STD, get_train_transforms, get_val_transforms
from src.models import CLASS_NAMES

ROOT = Path.cwd()
SPLITS = ("train", "val", "test")
failures: list[str] = []


def check(condition: bool, message: str) -> None:
    print(("PASS  " if condition else "FAIL  ") + message)
    if not condition:
        failures.append(message)


def main() -> int:
    frames = {name: pd.read_csv(ROOT / "data" / "splits" / f"{name}.csv") for name in SPLITS}
    config = json.loads((ROOT / "data" / "splits" / "split_config.json").read_text(encoding="utf-8"))
    print(f"Class order: {list(CLASS_NAMES)}; split seed {config['random_seed']}")

    for name, frame in frames.items():
        counts = frame["dx"].value_counts().to_dict()
        check(set(frame["dx"]) <= set(CLASS_NAMES), f"{name}: labels are all in the shared class mapping")
        check(set(CLASS_NAMES) <= set(frame["dx"]), f"{name}: all seven classes present")
        check(len(frame) == config["image_counts"][name], f"{name}: {len(frame)} rows match split_config.json")
        check(counts == config["class_distribution"][name], f"{name}: class counts match split_config.json")
        check(not frame[["image_path", "dx", "lesion_id", "image_id"]].isna().any().any(), f"{name}: no missing path/label/id values")
        check(frame["image_id"].is_unique, f"{name}: image_id values are unique")
        missing = [p for p in frame["image_path"] if not (ROOT / p).is_file()]
        check(not missing, f"{name}: every image path resolves ({len(missing)} missing, e.g. {missing[:2]})")

    lesions = {name: set(frame["lesion_id"]) for name, frame in frames.items()}
    images = {name: set(frame["image_id"]) for name, frame in frames.items()}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        check(not lesions[a] & lesions[b], f"no lesion_id overlap: {a} vs {b}")
        check(not images[a] & images[b], f"no image_id overlap: {a} vs {b}")

    metadata = pd.read_csv(ROOT / "data" / "raw" / "HAM10000_metadata.csv").set_index("image_id")["dx"]
    combined = pd.concat(frames.values())
    check(all(metadata[row.image_id] == row.dx for row in combined.itertuples()), "manifest labels agree with HAM10000_metadata.csv")

    unreadable = []
    for path in combined["image_path"]:
        try:
            with Image.open(ROOT / path) as image:
                image.verify()
        except Exception as error:  # report every failure instead of stopping at the first
            unreadable.append(f"{path}: {error}")
    check(not unreadable, f"all {len(combined)} images decode ({len(unreadable)} unreadable: {unreadable[:2]})")

    val_transform, train_transform = get_val_transforms(), get_train_transforms()
    sample = Image.open(ROOT / combined["image_path"].iloc[0]).convert("RGB")
    check(torch.equal(val_transform(sample), val_transform(sample)), "validation/test transform is deterministic")
    torch.manual_seed(0)
    differs = any(not torch.equal(train_transform(sample), train_transform(sample)) for _ in range(5))
    check(differs, "training augmentation transform is stochastic")
    check(tuple(val_transform(sample).shape) == (3, 224, 224), "validation transform yields 3x224x224")
    check("RandomHorizontalFlip" not in repr(val_transform) and "RandomRotation" not in repr(val_transform) and "ColorJitter" not in repr(val_transform),
          "validation transform contains no random operations")
    check(tuple(IMAGENET_MEAN) == (0.485, 0.456, 0.406) and tuple(IMAGENET_STD) == (0.229, 0.224, 0.225), "ImageNet normalization constants")

    print(f"\n{len(failures)} failed check(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

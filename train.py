"""Train VGG16 on a fixed HAM10000 split with configurable ablation settings."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from src.data.dataloader import get_dataloaders
from src.evaluation.metrics import classification_metrics
from src.losses import FocalLoss
from src.models import CLASS_NAMES, create_vgg16
from src.training import load_checkpoint, seed_everything, train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loss", choices=("cross_entropy", "focal"), default="cross_entropy")
    parser.add_argument("--augmentation", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0 if sys.platform == "win32" else 4)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--no-pretrained", action="store_true", help="Use random initialization; default downloads ImageNet weights if needed")
    parser.add_argument("--max-oom-restarts", type=int, default=2, help="Times to restart at half the batch size after a CUDA OOM")
    parser.add_argument("--experiment-name", help="Run tag; defaults to <loss>_aug-<true|false>")
    parser.add_argument("--output-dir", type=Path, help="Defaults to outputs/vgg16/<experiment>")
    parser.add_argument("--checkpoint-dir", type=Path, help="Defaults to checkpoints/vgg16/<experiment>")
    parser.add_argument("--smoke-samples", type=int, help="Use only the first N train and val samples (smoke test; never report as results)")
    parser.add_argument("--evaluate-test", action="store_true", help="Evaluate the selected checkpoint on the held-out test split. Use once per finalized protocol")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false; refusing CPU fallback")
    device = torch.device(args.device)
    experiment = args.experiment_name or f"{args.loss}_aug-{str(args.augmentation).lower()}"
    output_dir = args.output_dir or Path("outputs/vgg16") / experiment
    checkpoint_dir = args.checkpoint_dir or Path("checkpoints/vgg16") / experiment

    batch_size, oom_restarts = args.batch_size, []
    while True:
        try:
            run_experiment(args, device, batch_size, output_dir, checkpoint_dir, oom_restarts)
            return
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            oom_restarts.append(batch_size)
            smaller = batch_size // 2
            if smaller < 1 or len(oom_restarts) > args.max_oom_restarts:
                raise RuntimeError(
                    f"CUDA out of memory at batch sizes {oom_restarts}; stopping. No CPU fallback was attempted."
                ) from None
            print(f"CUDA out of memory at batch size {batch_size}; restarting the run from scratch at {smaller}.")
            batch_size = smaller


def run_experiment(args, device, batch_size, output_dir, checkpoint_dir, oom_restarts) -> None:
    seed_everything(args.seed)
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=Path.cwd(), batch_size=batch_size, num_workers=args.workers,
        pin_memory=device.type == "cuda", augmentation=args.augmentation,
    )
    if args.smoke_samples:
        train_loader = _first_samples(train_loader, args.smoke_samples, shuffle=True)
        val_loader = _first_samples(val_loader, args.smoke_samples, shuffle=False)
    model = create_vgg16(pretrained=not args.no_pretrained)
    criterion = build_loss(args.loss)
    config = {
        "model": "vgg16", "class_names": list(CLASS_NAMES), "loss": args.loss,
        "focal_gamma": 2.0 if args.loss == "focal" else None,
        "augmentation": args.augmentation, "seed": args.seed, "image_size": 224,
        "batch_size": batch_size, "oom_restarted_from_batch_sizes": oom_restarts,
        "epochs": args.epochs, "learning_rate": args.learning_rate,
        "optimizer": "Adam", "patience": args.patience, "device": str(device),
        "amp": device.type == "cuda", "workers": args.workers,
        "pretrained": not args.no_pretrained,
        "pretrained_weights": None if args.no_pretrained else "torchvision VGG16_Weights.IMAGENET1K_V1",
        "split_manifest": "data/splits/{train,val,test}.csv", "selected_by": "validation macro F1",
        "smoke_samples": args.smoke_samples, "evaluate_test": args.evaluate_test,
    }
    result, _ = train_model(
        model, train_loader, val_loader, criterion, device=device, output_dir=output_dir,
        config=config, epochs=args.epochs, learning_rate=args.learning_rate,
        patience=args.patience, amp=True, checkpoint_dir=checkpoint_dir,
    )
    payload = load_checkpoint(result["checkpoint"], model, device)
    report = {
        "status": "smoke_completed" if args.smoke_samples else "completed",
        "best_epoch": result["best_epoch"], "best_val_macro_f1": result["best_val_macro_f1"],
        "validation_metrics_at_best_epoch": payload["val_metrics"],
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "checkpoint_path": result["checkpoint"],
        "checkpoint_bytes": Path(result["checkpoint"]).stat().st_size,
        "training_seconds": result["training_seconds"], "config": config,
    }
    evaluated = payload["val_metrics"]
    split = "validation"
    if args.evaluate_test:
        test_loss, test_metrics = evaluate_once(model, test_loader, criterion, device)
        report.update(test_loss=test_loss, test_metrics=test_metrics)
        evaluated, split = test_metrics, "test"
        _save_confusion_plot(test_metrics["confusion_matrix"], output_dir / "confusion_matrix.png", "test")
    else:
        _save_confusion_plot(evaluated["confusion_matrix"], output_dir / "confusion_matrix.png", "validation")
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output_dir / "classification_report.json").write_text(
        json.dumps({"split": split, "per_class": evaluated["per_class"]}, indent=2), encoding="utf-8"
    )
    print(f"Saved run artifacts under {output_dir} and checkpoint under {checkpoint_dir} ({split} confusion matrix)")


def build_loss(name: str) -> torch.nn.Module:
    if name == "focal":
        return FocalLoss(gamma=2.0)
    return torch.nn.CrossEntropyLoss()


def _first_samples(loader: DataLoader, count: int, shuffle: bool) -> DataLoader:
    subset = Subset(loader.dataset, range(min(count, len(loader.dataset))))
    return DataLoader(subset, batch_size=loader.batch_size, shuffle=shuffle,
                      num_workers=loader.num_workers, pin_memory=loader.pin_memory)


def evaluate_once(model, loader, criterion, device):
    model.eval()
    total_loss, total = 0.0, 0
    targets, predictions = [], []
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, labels)
            total_loss += float(loss) * labels.size(0)
            total += labels.size(0)
            targets.extend(labels.cpu().tolist())
            predictions.extend(logits.argmax(1).cpu().tolist())
    if not total:
        raise ValueError("Test DataLoader is empty")
    return total_loss / total, classification_metrics(targets, predictions)


def _save_confusion_plot(matrix, path: Path, split: str) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    figure, axis = plt.subplots(figsize=(9, 7))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axis)
    axis.set(xlabel="Predicted", ylabel="True", title=f"VGG16 {split} confusion matrix")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()

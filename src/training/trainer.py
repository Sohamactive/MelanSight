"""Training and validation loop for reproducible classification experiments."""

from __future__ import annotations

import csv
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.evaluation.metrics import classification_metrics


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, amp=False):
    training = optimizer is not None
    model.train(training)
    loss_sum, count, targets_all, predictions_all = 0.0, 0, [], []
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                logits = model(images)
                loss = criterion(logits, targets)
            if training:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            batch_size = targets.size(0)
            loss_sum += float(loss.detach()) * batch_size
            count += batch_size
            predictions_all.extend(logits.detach().argmax(1).cpu().tolist())
            targets_all.extend(targets.detach().cpu().tolist())
    if not count:
        raise ValueError("Cannot run an epoch with an empty DataLoader")
    return loss_sum / count, classification_metrics(targets_all, predictions_all)


def evaluate(model, loader, criterion, device, amp: bool = False) -> tuple[float, dict]:
    return _run_epoch(model, loader, criterion, device, amp=amp)


def _write_history(history: list[dict], destination: Path) -> None:
    with destination.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)


def _atomic_torch_save(payload, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def load_checkpoint(path: str | Path, model: nn.Module, device: torch.device) -> dict:
    payload = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    return payload


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    *,
    device: torch.device,
    output_dir: str | Path,
    config: dict,
    epochs: int = 20,
    learning_rate: float = 1e-4,
    patience: int = 5,
    amp: bool = True,
    checkpoint_dir: str | Path | None = None,
) -> tuple[dict, list[dict]]:
    if epochs < 1 or patience < 1:
        raise ValueError("epochs and patience must be positive")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    model.to(device)
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=learning_rate)
    use_amp = bool(amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_score, best_epoch, stale_epochs, history = -1.0, 0, 0, []
    checkpoint = Path(checkpoint_dir) / "best.pt" if checkpoint_dir else out / "best.pt"
    training_started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        train_loss, train_metrics = _run_epoch(model, train_loader, criterion, device, optimizer, scaler, use_amp)
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device, use_amp)
        record = {
            "epoch": epoch, "train_loss": train_loss, "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_loss, "val_macro_f1": val_metrics["macro_f1"],
            "epoch_seconds": time.perf_counter() - started,
            "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        }
        history.append(record)
        _write_history(history, out / "history.csv")
        print(f"epoch {epoch}/{epochs} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_macro_f1={val_metrics['macro_f1']:.4f} ({record['epoch_seconds']:.1f}s)")
        if val_metrics["macro_f1"] > best_score:
            best_score, best_epoch, stale_epochs = val_metrics["macro_f1"], epoch, 0
            _atomic_torch_save({"model_state_dict": model.state_dict(), "epoch": epoch, "val_metrics": val_metrics, "config": config}, checkpoint)
        else:
            stale_epochs += 1
        if stale_epochs >= patience:
            print(f"early stopping after {epoch} epochs; best epoch {best_epoch}")
            break
    return {
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_score,
        "checkpoint": str(checkpoint),
        "training_seconds": time.perf_counter() - training_started,
    }, history

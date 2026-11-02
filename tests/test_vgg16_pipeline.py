from pathlib import Path

import pandas as pd
import pytest
import torch
from torch import nn

from src.data.dataloader import get_dataloaders
from src.losses import FocalLoss
from src.models import CLASS_NAMES, create_vgg16, get_gradcam_target_layer
from src.training import load_checkpoint, train_model

ROOT = Path(__file__).resolve().parents[1]
needs_images = pytest.mark.skipif(
    not (ROOT / "data" / "raw" / "HAM10000_metadata.csv").exists(),
    reason="HAM10000 images are not present under data/raw",
)


def test_class_order_is_the_shared_contract():
    assert CLASS_NAMES == ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")


def test_vgg16_outputs_raw_logits_for_seven_classes():
    model = create_vgg16(pretrained=False).eval()
    with torch.no_grad():
        logits = model(torch.zeros(2, 3, 224, 224))
    assert logits.shape == (2, 7)
    assert not torch.allclose(logits.sum(dim=1), torch.ones(2)), "model must not apply softmax"
    assert isinstance(model.classifier[6], nn.Linear) and model.classifier[6].out_features == 7


def test_gradcam_target_layer_is_last_convolution_with_expected_map_size():
    model = create_vgg16(pretrained=False).eval()
    layer = get_gradcam_target_layer(model)
    assert isinstance(layer, nn.Conv2d)
    captured = {}
    handle = layer.register_forward_hook(lambda _m, _i, out: captured.update(shape=tuple(out.shape)))
    with torch.no_grad():
        model(torch.zeros(1, 3, 224, 224))
    handle.remove()
    assert captured["shape"] == (1, 512, 14, 14)


@pytest.mark.parametrize("criterion", [nn.CrossEntropyLoss(), FocalLoss(gamma=2.0)])
def test_loss_is_finite_and_backpropagates(criterion):
    model = create_vgg16(pretrained=False)
    loss = criterion(model(torch.randn(2, 3, 224, 224)), torch.tensor([4, 5]))
    loss.backward()
    assert torch.isfinite(loss)
    assert model.classifier[6].weight.grad is not None


def test_split_manifests_do_not_share_lesions_or_images():
    frames = {name: pd.read_csv(ROOT / "data" / "splits" / f"{name}.csv") for name in ("train", "val", "test")}
    for first, second in (("train", "val"), ("train", "test"), ("val", "test")):
        assert not set(frames[first]["lesion_id"]) & set(frames[second]["lesion_id"])
        assert not set(frames[first]["image_id"]) & set(frames[second]["image_id"])
    for frame in frames.values():
        assert set(frame["dx"]) == set(CLASS_NAMES)


@needs_images
def test_dataloader_batch_labels_and_deterministic_eval_transform():
    train_loader, val_loader, _ = get_dataloaders(data_dir=ROOT, batch_size=4, num_workers=0, pin_memory=False, augmentation=False)
    images, labels = next(iter(train_loader))
    assert images.shape == (4, 3, 224, 224) and labels.dtype == torch.int64
    assert labels.min() >= 0 and labels.max() < len(CLASS_NAMES)
    first, _ = val_loader.dataset[0]
    second, _ = val_loader.dataset[0]
    assert torch.equal(first, second)


@needs_images
def test_tiny_training_run_saves_history_and_reloadable_checkpoint(tmp_path):
    train_loader, val_loader, _ = get_dataloaders(data_dir=ROOT, batch_size=4, num_workers=0, pin_memory=False, augmentation=False)
    subset = lambda loader: torch.utils.data.DataLoader(torch.utils.data.Subset(loader.dataset, range(8)), batch_size=4)  # noqa: E731
    model = create_vgg16(pretrained=False)
    result, history = train_model(
        model, subset(train_loader), subset(val_loader), nn.CrossEntropyLoss(),
        device=torch.device("cpu"), output_dir=tmp_path / "out", checkpoint_dir=tmp_path / "ckpt",
        config={"smoke": True}, epochs=1, patience=1, amp=False,
    )
    assert (tmp_path / "out" / "config.json").exists() and (tmp_path / "out" / "history.csv").exists()
    assert len(history) == 1 and torch.isfinite(torch.tensor(history[0]["train_loss"]))
    restored = create_vgg16(pretrained=False)
    payload = load_checkpoint(result["checkpoint"], restored, torch.device("cpu"))
    assert payload["epoch"] == 1
    for actual, expected in zip(restored.state_dict().values(), model.state_dict().values(), strict=True):
        torch.testing.assert_close(actual.cpu(), expected.cpu())

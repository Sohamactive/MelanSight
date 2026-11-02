import torch
from torch import nn

from src.evaluation.metrics import classification_metrics
from src.losses import FocalLoss
from src.models import CLASS_NAMES
from src.training import load_checkpoint


def test_focal_loss_gamma_zero_matches_cross_entropy():
    logits = torch.tensor([[1.0, 2.0, -1.0], [0.5, -0.2, 1.3]])
    targets = torch.tensor([1, 2])
    actual = FocalLoss(gamma=0)(logits, targets)
    expected = nn.CrossEntropyLoss()(logits, targets)
    torch.testing.assert_close(actual, expected)


def test_metrics_keep_fixed_class_order_and_melanoma_rates():
    target = list(range(7))
    metrics = classification_metrics(target, target)
    assert metrics["class_names"] == list(CLASS_NAMES)
    assert metrics["macro_f1"] == 1.0
    assert metrics["melanoma_sensitivity"] == 1.0
    assert metrics["melanoma_specificity"] == 1.0
    assert metrics["confusion_matrix"][4][4] == 1


def test_checkpoint_round_trip(tmp_path):
    original = nn.Linear(3, 7)
    checkpoint = tmp_path / "best.pt"
    torch.save({"model_state_dict": original.state_dict(), "epoch": 2}, checkpoint)
    restored = nn.Linear(3, 7)
    payload = load_checkpoint(checkpoint, restored, torch.device("cpu"))
    assert payload["epoch"] == 2
    for actual, expected in zip(restored.parameters(), original.parameters(), strict=True):
        torch.testing.assert_close(actual, expected)

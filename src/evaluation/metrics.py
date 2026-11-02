"""Metrics with fixed HAM10000 class order."""

import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, precision_recall_fscore_support

from src.models import CLASS_NAMES


def classification_metrics(targets, predictions) -> dict:
    labels = list(range(len(CLASS_NAMES)))
    matrix = confusion_matrix(targets, predictions, labels=labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, predictions, labels=labels, zero_division=0
    )
    mel_index = CLASS_NAMES.index("mel")
    tp = int(matrix[mel_index, mel_index])
    fn = int(matrix[mel_index, :].sum() - tp)
    fp = int(matrix[:, mel_index].sum() - tp)
    tn = int(matrix.sum() - tp - fn - fp)
    return {
        "class_names": list(CLASS_NAMES),
        "macro_f1": float(np.mean(f1)),
        "balanced_accuracy": float(balanced_accuracy_score(targets, predictions)),
        "per_class": {
            name: {"precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i]), "support": int(support[i])}
            for i, name in enumerate(CLASS_NAMES)
        },
        "melanoma_sensitivity": float(tp / (tp + fn)) if tp + fn else 0.0,
        "melanoma_specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "confusion_matrix": matrix.tolist(),
    }

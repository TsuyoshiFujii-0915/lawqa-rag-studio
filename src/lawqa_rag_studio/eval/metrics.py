"""Evaluation metrics utilities."""
from __future__ import annotations

from typing import Sequence


def accuracy(preds: Sequence[int], labels: Sequence[int]) -> float:
    """Compute accuracy score.

    Args:
        preds: Predicted labels.
        labels: Ground truth labels.

    Returns:
        Accuracy in range [0, 1].
    """
    if not preds:
        return 0.0
    correct = sum(int(p == l) for p, l in zip(preds, labels))
    return correct / len(preds)


def macro_f1(preds: Sequence[int], labels: Sequence[int], num_classes: int | None = None) -> float:
    """Compute macro F1 score (placeholder implementation).

    Args:
        preds: Predicted labels.
        labels: Ground truth labels.
        num_classes: Optional number of classes.

    Returns:
        Macro F1 score.
    """
    if not preds:
        return 0.0
    classes = range(num_classes) if num_classes is not None else set(labels) | set(preds)
    f1s: list[float] = []
    for c in classes:
        tp = sum(1 for p, l in zip(preds, labels) if p == c and l == c)
        fp = sum(1 for p, l in zip(preds, labels) if p == c and l != c)
        fn = sum(1 for p, l in zip(preds, labels) if p != c and l == c)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        f1s.append(f1)
    return sum(f1s) / len(f1s)


__all__ = ["accuracy", "macro_f1"]

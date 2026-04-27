"""Evaluation: accuracy, per-class F1, macro-F1, and confusion matrix."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from config import LABELS


def compute_metrics(true_labels: list[int], pred_labels: list[int]) -> dict:
    """Return accuracy, per-class F1, and macro-F1."""
    return {
        "accuracy": accuracy_score(true_labels, pred_labels),
        "macro_f1": f1_score(true_labels, pred_labels, average="macro", zero_division=0),
        "per_class_f1": f1_score(
            true_labels, pred_labels, average=None, labels=list(range(len(LABELS))), zero_division=0
        ).tolist(),
    }


def print_report(true_labels: list[int], pred_labels: list[int]) -> None:
    print(classification_report(true_labels, pred_labels, target_names=LABELS, zero_division=0))


def save_confusion_matrix(true_labels: list[int], pred_labels: list[int], path: Path) -> None:
    cm = confusion_matrix(true_labels, pred_labels, labels=list(range(len(LABELS))))
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=LABELS, yticklabels=LABELS, ax=ax, cmap="Blues")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

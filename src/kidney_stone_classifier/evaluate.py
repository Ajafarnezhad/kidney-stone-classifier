"""Test-set evaluation and reporting for classification models."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe: never assume a display is attached
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def evaluate_classification_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    model_name: str,
    output_dir: Path,
) -> dict:
    """Evaluate on the test set and save a classification report + confusion matrix.

    Saves artifacts to disk instead of calling `plt.show()`, which silently
    does nothing in a non-interactive/headless run.
    """
    model.eval()
    all_preds: list[float] = []
    all_labels: list[float] = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs).squeeze(-1)
            preds = (torch.sigmoid(outputs) > 0.5).float().cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    report = classification_report(all_labels, all_preds, target_names=["Normal", "Stone"], output_dict=True)
    logger.info(
        "Classification report for %s:\n%s",
        model_name,
        classification_report(all_labels, all_preds, target_names=["Normal", "Stone"]),
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / f"{model_name}_report.json").write_text(json.dumps(report, indent=2))

    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Stone"], yticklabels=["Normal", "Stone"], ax=ax)
    ax.set_title(f"Confusion Matrix - {model_name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.savefig(output_dir / f"{model_name}_confusion_matrix.png", bbox_inches="tight")
    plt.close(fig)

    return report


def plot_training_history(history, model_name: str, output_dir: Path) -> Path:
    epochs = range(1, len(history.train_loss) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].plot(epochs, history.train_loss, label="Train Loss")
    axes[0].plot(epochs, history.val_loss, label="Val Loss")
    axes[0].set_title(f"{model_name} Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, history.train_acc, label="Train Acc")
    axes[1].plot(epochs, history.val_acc, label="Val Acc")
    axes[1].set_title(f"{model_name} Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    axes[2].plot(epochs, history.val_auroc, label="Val AUROC")
    axes[2].set_title(f"{model_name} AUROC")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()

    fig.tight_layout()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{model_name}_history.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path

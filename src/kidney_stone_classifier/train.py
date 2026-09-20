"""Training loop for a single classification model."""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torcheval.metrics import BinaryAUROC

from .models import build_classification_model

logger = logging.getLogger(__name__)


@dataclass
class TrainingHistory:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    train_acc: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)
    val_auroc: list[float] = field(default_factory=list)


def _run_validation(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device):
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    all_probs: list[float] = []
    all_labels: list[float] = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device).float()
            with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                outputs = model(inputs).squeeze(-1)
                loss = criterion(outputs, labels)
            probs = torch.sigmoid(outputs)

            val_loss += loss.item() * inputs.size(0)
            val_correct += int(((probs > 0.5).float() == labels).sum().item())
            val_total += inputs.size(0)
            all_probs.extend(probs.detach().cpu().tolist())
            all_labels.extend(labels.detach().cpu().tolist())

    auroc_metric = BinaryAUROC()
    auroc_metric.update(torch.tensor(all_probs), torch.tensor(all_labels))
    return val_loss / val_total, val_correct / val_total, auroc_metric.compute().item()


def train_one_model(
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    save_dir: Path,
) -> tuple[nn.Module, TrainingHistory, Path]:
    """Train a single backbone and return the best (by val accuracy) weights.

    Uses the current `torch.autocast` / `torch.amp.GradScaler` APIs, guarded
    so the same code path runs correctly on CPU (mixed precision disabled)
    or GPU.
    """
    model = build_classification_model(model_name).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda"))

    best_acc = 0.0
    best_state = copy.deepcopy(model.state_dict())
    history = TrainingHistory()

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_path = save_dir / f"{model_name}_best.pth"

    for epoch in range(epochs):
        start = time.time()
        model.train()
        running_loss, running_correct, total = 0.0, 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device).float()
            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                outputs = model(inputs).squeeze(-1)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * inputs.size(0)
            running_correct += int(((torch.sigmoid(outputs) > 0.5).float() == labels).sum().item())
            total += inputs.size(0)

        train_loss = running_loss / total
        train_acc = running_correct / total
        val_loss, val_acc, val_auroc = _run_validation(model, val_loader, criterion, device)

        history.train_loss.append(train_loss)
        history.train_acc.append(train_acc)
        history.val_loss.append(val_loss)
        history.val_acc.append(val_acc)
        history.val_auroc.append(val_auroc)

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, best_path)

        scheduler.step()
        logger.info(
            "[%s] epoch %d/%d - train_loss=%.4f train_acc=%.4f val_loss=%.4f val_acc=%.4f val_auroc=%.4f (%.1fs)",
            model_name,
            epoch + 1,
            epochs,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            val_auroc,
            time.time() - start,
        )

    model.load_state_dict(best_state)
    return model, history, best_path

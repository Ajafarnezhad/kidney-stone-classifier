"""Classification model factory (binary head over pretrained backbones)."""

from __future__ import annotations

import torch.nn as nn
import torchvision.models as tv_models
from efficientnet_pytorch import EfficientNet

SUPPORTED_MODELS = ("efficientnet", "resnet", "densenet")


def build_classification_model(model_name: str) -> nn.Module:
    """Build a pretrained backbone with a single-logit binary classification head."""
    if model_name == "efficientnet":
        model = EfficientNet.from_pretrained("efficientnet-b3")
        model._fc = nn.Linear(model._fc.in_features, 1)
    elif model_name == "resnet":
        model = tv_models.resnet101(weights="IMAGENET1K_V1")
        model.fc = nn.Linear(model.fc.in_features, 1)
    elif model_name == "densenet":
        model = tv_models.densenet161(weights="IMAGENET1K_V1")
        model.classifier = nn.Linear(model.classifier.in_features, 1)
    else:
        raise ValueError(f"Unsupported model '{model_name}'. Choose from {SUPPORTED_MODELS}.")
    return model

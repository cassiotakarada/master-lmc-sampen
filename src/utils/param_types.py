import json
import os
from typing import Dict

import torch.nn as nn

from .logging import get_logger

CATEGORY_MAP = {
    nn.Conv1d: "Convolutional",
    nn.Conv2d: "Convolutional",
    nn.Conv3d: "Convolutional",
    nn.Linear: "Linear",
    nn.Embedding: "Embedding",
    nn.BatchNorm1d: "BatchNorm",
    nn.BatchNorm2d: "BatchNorm",
    nn.BatchNorm3d: "BatchNorm",
    nn.ReLU: "Activation",
    nn.LeakyReLU: "Activation",
    nn.Sigmoid: "Activation",
    nn.Tanh: "Activation",
    nn.Softmax: "Activation",
    nn.MaxPool1d: "Pooling",
    nn.MaxPool2d: "Pooling",
    nn.MaxPool3d: "Pooling",
    nn.AvgPool1d: "Pooling",
    nn.AvgPool2d: "Pooling",
    nn.AvgPool3d: "Pooling",
    nn.AdaptiveAvgPool1d: "Pooling",
    nn.AdaptiveAvgPool2d: "Pooling",
    nn.AdaptiveAvgPool3d: "Pooling",
    nn.Flatten: "Flatten",
}


def _classify_module(module: nn.Module) -> str:
    for cls, name in CATEGORY_MAP.items():
        if isinstance(module, cls):
            return name
    return "Other"


def build_param_types(model: nn.Module) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for module_name, module in model.named_modules():
        for param_name, _ in module.named_parameters(recurse=False):
            full_name = f"{module_name}.{param_name}" if module_name else param_name
            mapping[full_name] = _classify_module(module)
    return mapping


def save_param_types(model: nn.Module, path: str, logger=None) -> None:
    logger = logger or get_logger()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    mapping = build_param_types(model)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)
    logger.info("Saved param_types mapping to %s (%d entries)", path, len(mapping))

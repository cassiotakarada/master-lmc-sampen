from typing import Any, Dict

import torch.nn as nn
from monai.networks.nets import DenseNet121, resnet18


def build_model(model_name: str, in_channels: int, num_classes: int) -> nn.Module:
    name = model_name.lower()
    if name == "densenet121":
        return DenseNet121(spatial_dims=2, in_channels=in_channels, out_channels=num_classes)
    if name == "resnet18":
        return resnet18(spatial_dims=2, n_input_channels=in_channels, num_classes=num_classes)
    raise ValueError(f"Unsupported model: {model_name}")

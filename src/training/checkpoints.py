import json
import os
from typing import Dict

import torch

from ..utils.logging import get_logger


def _log_size(path: str, logger) -> None:
    try:
        size = os.path.getsize(path)
        logger.info("Saved checkpoint: %s (%.2f KB)", path, size / 1024)
    except OSError:
        logger.warning("Checkpoint path not found after save: %s", path)


def save_initial_weights(model, path: str, logger=None) -> None:
    logger = logger or get_logger()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    _log_size(path, logger)


def save_epoch_weights(model, path: str, logger=None) -> None:
    logger = logger or get_logger()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    _log_size(path, logger)


def save_best_weights(model, path: str, logger=None) -> None:
    logger = logger or get_logger()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    _log_size(path, logger)


def save_config(config: Dict, path: str, logger=None) -> None:
    logger = logger or get_logger()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    logger.info("Saved config snapshot to %s", path)

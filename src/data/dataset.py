import os
from typing import List, Dict, Tuple, Optional

import torch
from monai.data import Dataset, CacheDataset, DataLoader

from ..utils.logging import get_logger


def _filter_missing(samples: List[Dict], logger) -> Tuple[List[Dict], List[str]]:
    filtered = []
    missing_paths: List[str] = []
    for sample in samples:
        if os.path.exists(sample["image"]):
            filtered.append(sample)
        else:
            missing_paths.append(sample["image"])
    if missing_paths:
        log_count = min(10, len(missing_paths))
        for path in missing_paths[:log_count]:
            logger.warning("Missing image file skipped: %s", path)
    return filtered, missing_paths


def create_datasets(train_samples: List[Dict], val_samples: List[Dict], transforms, cache_rate: float = 0.0, sample_log_count: int = 5, logger=None):
    logger = logger or get_logger()
    total_input = len(train_samples) + len(val_samples)
    train_samples, missing_train_paths = _filter_missing(train_samples, logger)
    val_samples, missing_val_paths = _filter_missing(val_samples, logger)
    missing_total = len(missing_train_paths) + len(missing_val_paths)
    if missing_total > 0:
        logger.warning("Skipped %d samples due to missing files", missing_total)
        if total_input > 0 and missing_total / total_input > 0.05:
            logger.warning("More than 5%% of samples are missing (%.2f%%)", (missing_total / total_input) * 100)

    for sample in train_samples[:sample_log_count]:
        logger.debug("Train sample post-filter | image=%s label=%s", sample.get("image"), sample.get("label"))
    for sample in val_samples[:sample_log_count]:
        logger.debug("Val sample post-filter | image=%s label=%s", sample.get("image"), sample.get("label"))

    DatasetCls = CacheDataset if cache_rate > 0 else Dataset
    logger.info("Building datasets with cache_rate=%.2f using %s", cache_rate, DatasetCls.__name__)
    if DatasetCls is CacheDataset:
        train_ds = DatasetCls(data=train_samples, transform=transforms, cache_rate=cache_rate) if train_samples else None
        val_ds = DatasetCls(data=val_samples, transform=transforms, cache_rate=cache_rate) if val_samples else None
    else:
        train_ds = DatasetCls(data=train_samples, transform=transforms) if train_samples else None
        val_ds = DatasetCls(data=val_samples, transform=transforms) if val_samples else None
    logger.info("Dataset sizes: train=%d val=%d", len(train_samples), len(val_samples))
    return train_ds, val_ds


def create_loaders(train_ds, val_ds, batch_size: int, num_workers: int, logger=None):
    logger = logger or get_logger()
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
    }
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs) if train_ds is not None else None
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs) if val_ds is not None else None
    logger.info("DataLoaders created: train=%s val=%s batch_size=%d num_workers=%d", bool(train_loader), bool(val_loader), batch_size, num_workers)
    return train_loader, val_loader

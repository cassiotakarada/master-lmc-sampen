import argparse
import logging
import os
import sys
import time
from typing import List

import monai
import numpy as np
import torch

from .config import TrainConfig
from .data.transforms import build_transforms
from .data.dataset import create_loaders
from .training.model import build_model
from .training.trainer import Trainer
from .training.checkpoints import save_initial_weights, save_config
from .utils.param_types import save_param_types
from .utils.paths import ensure_dir
from .utils.seed import set_seed
from .utils.logging import setup_logger, get_logger
from monai.apps import MedNISTDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MONAI 2D classification trainer")
    parser.add_argument("--db_path", type=str, default=None)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--results_dir", type=str, default=None)
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--cache_rate", type=float, default=None)
    parser.add_argument("--in_channels", type=int, default=None)
    parser.add_argument("--num_classes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--save_every_epoch", action="store_true", default=None)
    parser.add_argument("--val_split", type=float, default=None)
    parser.add_argument("--model_name", type=str, default="densenet121")
    parser.add_argument("--log_level", type=str, default=None)
    parser.add_argument("--debug", action="store_true", default=None)
    parser.add_argument("--sample_log_count", type=int, default=None)
    return parser.parse_args()


def merge_config(args: argparse.Namespace) -> TrainConfig:
    cfg = TrainConfig()
    for field in cfg.__dataclass_fields__:  # type: ignore[attr-defined]
        arg_val = getattr(args, field, None)
        if arg_val is not None:
            setattr(cfg, field, arg_val)
    if getattr(args, "debug", None):
        cfg.debug = True
        cfg.log_level = "DEBUG"
    if getattr(args, "log_level", None):
        cfg.log_level = args.log_level
    return cfg


def log_dataset_samples(dataset, split: str, n: int) -> None:
    logger = get_logger()
    preview = min(n, len(dataset))
    for idx in range(preview):
        item = dataset[idx]
        logger.info("Sample %s | idx=%d label=%s shape=%s", split, idx, item.get("label"), np.array(item.get("image")).shape)


def prepare_mednist_datasets(cfg: TrainConfig, transforms):
    logger = get_logger()
    ensure_dir(cfg.data_root)
    train_ds = MedNISTDataset(
        root_dir=cfg.data_root,
        section="training",
        download=True,
        transform=transforms,
        cache_rate=cfg.cache_rate,
        num_workers=cfg.num_workers,
    )
    val_ds = MedNISTDataset(
        root_dir=cfg.data_root,
        section="validation",
        download=True,
        transform=transforms,
        cache_rate=cfg.cache_rate,
        num_workers=cfg.num_workers,
    )
    logger.info("Loaded MedNIST: train=%d val=%d", len(train_ds), len(val_ds))
    return train_ds, val_ds


def main() -> None:
    start_time = time.perf_counter()
    args = parse_args()
    cfg = merge_config(args)
    set_seed(cfg.seed)

    run_dir_path = os.path.join(cfg.results_dir, cfg.run_id)
    existed = os.path.exists(run_dir_path)
    run_dir = ensure_dir(run_dir_path)
    logger = setup_logger(run_dir, level="DEBUG" if cfg.debug else cfg.log_level)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(
        "Startup: python=%s torch=%s monai=%s device=%s seed=%d log_level=%s",
        sys.version.replace("\n", " "),
        torch.__version__,
        monai.__version__,
        device,
        cfg.seed,
        logging.getLevelName(logger.level),
    )
    logger.info("Run directory: %s (existed=%s)", run_dir, existed)
    logger.info("Configuration: %s", cfg.to_dict())

    try:
        transforms = build_transforms()
        logger.info("Transform pipeline: %s", transforms)
        logger.info("Expected modality: grayscale in_channels=%d", cfg.in_channels)

        train_ds, val_ds = prepare_mednist_datasets(cfg, transforms)
        log_dataset_samples(train_ds, "train", cfg.sample_log_count)
        log_dataset_samples(val_ds, "val", cfg.sample_log_count)

        train_loader, val_loader = create_loaders(train_ds, val_ds, batch_size=cfg.batch_size, num_workers=cfg.num_workers, logger=logger)

        model = build_model(args.model_name, in_channels=cfg.in_channels, num_classes=cfg.num_classes)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        has_nan = any(torch.isnan(p).any().item() for p in model.parameters())
        logger.info(
            "Model: %s in_channels=%d num_classes=%d total_params=%d trainable_params=%d has_nan_init=%s",
            args.model_name,
            cfg.in_channels,
            cfg.num_classes,
            total_params,
            trainable_params,
            has_nan,
        )

        save_initial_weights(model, os.path.join(run_dir, "initial_weights.pth"), logger=logger)
        save_config(cfg.to_dict(), os.path.join(run_dir, "config.json"), logger=logger)
        save_param_types(model, os.path.join(run_dir, "param_types.json"), logger=logger)

        trainer = Trainer(
            model=model,
            device=device,
            train_loader=train_loader,
            val_loader=val_loader,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            epochs=cfg.epochs,
            results_dir=run_dir,
            save_every_epoch=cfg.save_every_epoch,
            logger=logger,
        )

        trainer.fit()
        if trainer.best_val_acc is not None and cfg.epochs >= 50 and trainer.best_val_acc < 0.30:
            logger.warning("Validation accuracy is below 0.30 after %d epochs. Consider reviewing transforms, learning rate, or batch size.", cfg.epochs)
        logger.info(
            "Run complete: best_val_loss=%s best_val_acc=%s history=%s param_types=%s best_weights=%s",
            trainer.best_val_loss,
            trainer.best_val_acc,
            os.path.join(run_dir, "history.csv"),
            os.path.join(run_dir, "param_types.json"),
            os.path.join(run_dir, "best_weights.pth"),
        )
        elapsed = time.perf_counter() - start_time
        logger.info("Total runtime: %.2fs", elapsed)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Training failed: %s", exc)
        fail_path = os.path.join(run_dir, "FAILED.txt")
        with open(fail_path, "w", encoding="utf-8") as f:
            f.write(f"Training failed: {exc}\n")
        raise


if __name__ == "__main__":
    main()

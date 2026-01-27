import csv
import os
import time
from typing import Optional, Dict

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .metrics import accuracy
from .checkpoints import save_epoch_weights, save_best_weights
from ..utils.logging import get_logger


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        train_loader: Optional[DataLoader],
        val_loader: Optional[DataLoader],
        lr: float,
        weight_decay: float,
        epochs: int,
        results_dir: str,
        save_every_epoch: bool,
        logger=None,
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = nn.CrossEntropyLoss()
        self.results_dir = results_dir
        self.save_every_epoch = save_every_epoch
        self.best_val_loss: Optional[float] = None
        self.best_val_acc: Optional[float] = None
        self.history_path = os.path.join(results_dir, "history.csv")
        self.logger = logger or get_logger()
        self.logged_first_batch = False
        os.makedirs(results_dir, exist_ok=True)
        self._init_history()

    def _init_history(self) -> None:
        with open(self.history_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss", "val_accuracy"])
        self.logger.info("Initialized history log at %s", self.history_path)

    def train_epoch(self) -> float:
        if self.train_loader is None:
            return 0.0
        self.model.train()
        total_loss = 0.0
        for batch_idx, batch in enumerate(self.train_loader):
            images = batch["image"].to(self.device)
            labels = batch["label"].long().to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.criterion(logits, labels)
            if not torch.isfinite(loss):
                self.logger.error(
                    "Non-finite loss detected at batch %d | shape=%s min=%.4f max=%.4f labels_min=%s labels_max=%s",
                    batch_idx,
                    tuple(images.shape),
                    images.min().item(),
                    images.max().item(),
                    labels.min().item(),
                    labels.max().item(),
                )
                raise RuntimeError("Loss became non-finite, aborting training")
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            if not self.logged_first_batch:
                self._log_first_batch(images, labels)
                self.logged_first_batch = True
        return total_loss / max(1, len(self.train_loader))

    def validate_epoch(self) -> Dict[str, float]:
        if self.val_loader is None:
            return {"val_loss": 0.0, "val_accuracy": 0.0}
        self.model.eval()
        total_loss = 0.0
        total_acc = 0.0
        with torch.no_grad():
            for batch in self.val_loader:
                images = batch["image"].to(self.device)
                labels = batch["label"].long().to(self.device)
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                total_acc += accuracy(logits, labels)
        num_batches = max(1, len(self.val_loader))
        return {"val_loss": total_loss / num_batches, "val_accuracy": total_acc / num_batches}

    def fit(self) -> None:
        for epoch in range(1, self.epochs + 1):
            start = time.perf_counter()
            train_loss = self.train_epoch()
            val_metrics = self.validate_epoch()
            val_loss = val_metrics["val_loss"]
            val_acc = val_metrics["val_accuracy"]
            self._log_epoch(epoch, train_loss, val_loss, val_acc)
            duration = time.perf_counter() - start
            lr = self.optimizer.param_groups[0]["lr"]
            gpu_mem = torch.cuda.memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0
            self.logger.info(
                "Epoch %d | lr=%.6f | train_loss=%.4f | val_loss=%.4f | val_acc=%.4f | time=%.2fs | gpu_mem=%.1fMB",
                epoch,
                lr,
                train_loss,
                val_loss,
                val_acc,
                duration,
                gpu_mem,
            )
            if self.save_every_epoch:
                save_epoch_weights(self.model, os.path.join(self.results_dir, f"epoch_{epoch:03d}.pth"), logger=self.logger)
            if self.best_val_loss is None or val_loss < self.best_val_loss:
                old_best = self.best_val_loss
                self.best_val_loss = val_loss
                self.best_val_acc = val_acc
                self.logger.info("New best val loss: %s -> %s", old_best, val_loss)
                save_best_weights(self.model, os.path.join(self.results_dir, "best_weights.pth"), logger=self.logger)

    def _log_epoch(self, epoch: int, train_loss: float, val_loss: float, val_acc: float) -> None:
        try:
            with open(self.history_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([epoch, train_loss, val_loss, val_acc])
            self.logger.debug("Appended history for epoch %d", epoch)
        except OSError as exc:  # noqa: PERF203
            self.logger.error("Failed to write history for epoch %d: %s", epoch, exc)

    def _log_first_batch(self, images: torch.Tensor, labels: torch.Tensor) -> None:
        labels_np = labels.detach().cpu()
        labels_list = labels_np.tolist()
        max_label = int(labels_np.max().item()) if labels_np.numel() > 0 else 0
        hist = labels_np.bincount(minlength=max_label + 1 if max_label >= 0 else 1).tolist()
        self.logger.info(
            "First batch stats | shape=%s dtype=%s min=%.4f max=%.4f labels=%s hist=%s",
            tuple(images.shape),
            images.dtype,
            images.min().item(),
            images.max().item(),
            labels_list,
            hist,
        )

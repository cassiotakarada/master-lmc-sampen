import csv
import json
import os
import time
from typing import Optional, Dict

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .metrics import accuracy
from .checkpoints import save_epoch_weights, save_best_weights
from .complexity_monitor import ComplexityMonitor
from .overfit import detect_overfit_onset, epocas_sem_melhora
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
        tb_log_dir: Optional[str] = None,
        test_loader: Optional[DataLoader] = None,
        monitor: Optional[ComplexityMonitor] = None,
        patience: int = 3,
        overfit_delta: float = 0.0,
        early_stop: bool = False,
        eval_train: bool = False,
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        # O conjunto de TESTE e olhado uma unica vez, em evaluate_test(), no fim do
        # treino e com os pesos da melhor epoca. Ele NUNCA aparece em fit().
        self.test_loader = test_loader
        self.epochs = epochs
        self.optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = nn.CrossEntropyLoss()
        self.results_dir = results_dir
        self.save_every_epoch = save_every_epoch
        self.best_val_loss: Optional[float] = None
        self.best_val_acc: Optional[float] = None
        self.best_epoch: Optional[int] = None
        self.test_metrics: Optional[Dict[str, float]] = None
        # Monitor de complexidade ao vivo (F4). None = desligado.
        self.monitor = monitor
        self.patience = patience
        self.overfit_delta = overfit_delta
        # Desligado por padrao DE PROPOSITO: os experimentos do projeto treinam ate o fim
        # para ter a trajetoria completa de LMC/SampEn, e porque parar cedo mataria um
        # eventual caso de grokking (melhora tardia). Ligue so para economizar maquina.
        self.early_stop = early_stop
        self.parou_cedo: bool = False
        # Checagem anti-artefato: acuracia do PROPRIO treino em eval(). Overfitting real
        # = treino alto E validacao baixa. Artefato de BatchNorm = os dois baixos.
        self.eval_train = eval_train
        self._val_losses: list = []
        self.complexity_path = os.path.join(results_dir, "complexity_live.csv")
        self._complexity_header_written = False
        self._overfit_saved: bool = False
        self.history_path = os.path.join(results_dir, "history.csv")
        self.logger = logger or get_logger()
        self.logged_first_batch = False
        self.writer: Optional[SummaryWriter] = SummaryWriter(log_dir=tb_log_dir) if tb_log_dir else None
        os.makedirs(results_dir, exist_ok=True)
        self._init_history()

    def _init_history(self) -> None:
        colunas = ["epoch", "train_loss", "val_loss", "val_accuracy"]
        if self.eval_train:
            colunas += ["train_eval_loss", "train_eval_accuracy"]
        if self.monitor is not None:
            colunas += ["lmc_dense", "sampen_dense", "status"]
        self._history_cols = colunas
        with open(self.history_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(colunas)
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

    def _evaluate(self, loader: DataLoader) -> Dict[str, float]:
        """Passada de avaliacao sem gradiente. Compartilhada por validacao e teste,
        para que as duas metricas sejam calculadas exatamente do mesmo jeito."""
        self.model.eval()
        total_loss = 0.0
        total_acc = 0.0
        n_amostras = 0
        with torch.no_grad():
            for batch in loader:
                images = batch["image"].to(self.device)
                labels = batch["label"].long().to(self.device)
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                total_acc += accuracy(logits, labels)
                n_amostras += labels.numel()
        num_batches = max(1, len(loader))
        return {"loss": total_loss / num_batches, "accuracy": total_acc / num_batches,
                "n_amostras": float(n_amostras)}

    def validate_epoch(self) -> Dict[str, float]:
        if self.val_loader is None:
            return {"val_loss": 0.0, "val_accuracy": 0.0}
        m = self._evaluate(self.val_loader)
        return {"val_loss": m["loss"], "val_accuracy": m["accuracy"]}

    def evaluate_test(self) -> Optional[Dict[str, float]]:
        """Avalia o conjunto de TESTE UMA UNICA VEZ, com os pesos da melhor epoca.

        Regra metodologica (D5/D6 do plan.md): o teste nunca e consultado durante o
        treino, e os pesos usados aqui sao os da epoca de menor val_loss -- nunca os
        da ultima epoca, que ja pode estar em regime de overfitting.

        Devolve None se nao houver conjunto de teste. Grava test_metrics.json.
        """
        if self.test_loader is None:
            self.logger.warning("Sem conjunto de teste: avaliacao final pulada")
            return None

        melhor = os.path.join(self.results_dir, "bef_weights.pth")
        if os.path.isfile(melhor):
            estado = torch.load(melhor, map_location=self.device, weights_only=True)
            self.model.load_state_dict(estado)
            self.logger.info("Pesos da melhor epoca (%s) carregados para o teste: %s",
                             self.best_epoch, melhor)
        else:
            self.logger.warning("bef_weights.pth ausente; testando com os pesos atuais")

        m = self._evaluate(self.test_loader)
        self.test_metrics = {
            "test_loss": m["loss"],
            "test_accuracy": m["accuracy"],
            "n_amostras_teste": int(m["n_amostras"]),
            "epoca_avaliada": self.best_epoch,
            "best_val_loss": self.best_val_loss,
            "best_val_accuracy": self.best_val_acc,
            "observacao": ("Conjunto de teste avaliado UMA UNICA VEZ, no fim do treino, "
                           "com os pesos da epoca de menor val_loss."),
        }
        destino = os.path.join(self.results_dir, "test_metrics.json")
        with open(destino, "w", encoding="utf-8") as f:
            json.dump(self.test_metrics, f, indent=2, ensure_ascii=False)
        self.logger.info(
            "TESTE (uma unica avaliacao) | epoca=%s | test_loss=%.4f | test_acc=%.4f | n=%d -> %s",
            self.best_epoch, m["loss"], m["accuracy"], int(m["n_amostras"]), destino,
        )
        return self.test_metrics

    def fit(self) -> None:
        for epoch in range(1, self.epochs + 1):
            start = time.perf_counter()
            train_loss = self.train_epoch()
            val_metrics = self.validate_epoch()
            val_loss = val_metrics["val_loss"]
            val_acc = val_metrics["val_accuracy"]
            self._val_losses.append(val_loss)
            tr_eval = self._evaluate(self.train_loader) if (self.eval_train and self.train_loader) else None
            comp = None
            if self.monitor is not None:
                t_comp = time.perf_counter()
                comp = self.monitor.on_epoch_end(self.model, epoch, val_loss, val_acc)
                comp["overhead_s"] = time.perf_counter() - t_comp
                self._log_complexity(comp)
            self._log_epoch(epoch, train_loss, val_loss, val_acc, comp, tr_eval)
            duration = time.perf_counter() - start
            lr = self.optimizer.param_groups[0]["lr"]
            gpu_mem = torch.cuda.memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0
            extra = f" | {self.monitor.resumo_log(comp)}" if comp else ""
            if tr_eval:
                gap = tr_eval["accuracy"] - val_acc
                extra = (f" | train_eval_acc={tr_eval['accuracy']:.4f} gap={gap:+.4f}") + extra
            self.logger.info(
                "Epoch %d | lr=%.6f | train_loss=%.4f | val_loss=%.4f | val_acc=%.4f | time=%.2fs | gpu_mem=%.1fMB%s",
                epoch,
                lr,
                train_loss,
                val_loss,
                val_acc,
                duration,
                gpu_mem,
                extra,
            )
            if self.writer:
                self.writer.add_scalar("Loss/train", train_loss, epoch)
                self.writer.add_scalar("Loss/val", val_loss, epoch)
                self.writer.add_scalar("Accuracy/val", val_acc, epoch)
                if comp:
                    self.writer.add_scalar("Complexity/LMC_dense", comp["lmc"], epoch)
                    if comp["sampen"] == comp["sampen"]:      # nao-NaN
                        self.writer.add_scalar("Complexity/SampEn_dense", comp["sampen"], epoch)
                    if comp.get("sampen2d") == comp.get("sampen2d") and "sampen2d" in comp:
                        self.writer.add_scalar("Complexity/SampEn2D_dense", comp["sampen2d"], epoch)
            if self.save_every_epoch:
                save_epoch_weights(self.model, os.path.join(self.results_dir, f"epoch_{epoch:03d}.pth"), logger=self.logger)
            if self.best_val_loss is None or val_loss < self.best_val_loss:
                old_best = self.best_val_loss
                self.best_val_loss = val_loss
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                self._overfit_saved = False
                self.logger.info("New best val loss: %s -> %s", old_best, val_loss)
                save_best_weights(self.model, os.path.join(self.results_dir, "bef_weights.pth"), logger=self.logger)
            elif not self._overfit_saved:
                self._overfit_saved = True
                save_epoch_weights(self.model, os.path.join(self.results_dir, "aft_weights.pth"), logger=self.logger)
                self.logger.info("Saved aft_weights at epoch %d (first val_loss increase after best)", epoch)
            if self.early_stop:
                sem_melhora = epocas_sem_melhora(
                    [h["val_loss"] for h in self.monitor.historico] if self.monitor
                    else self._val_losses
                )
                if sem_melhora >= self.patience:
                    self.parou_cedo = True
                    self.logger.info(
                        "Early stopping na epoca %d: val_loss sem melhorar ha %d epocas "
                        "(patience=%d). ATENCAO: parar cedo impede observar grokking.",
                        epoch, sem_melhora, self.patience,
                    )
                    break

        self._salvar_relatorio_overfit()
        if self.writer:
            self.writer.close()

    def _log_complexity(self, comp: Dict) -> None:
        """Grava complexity_live.csv, com todas as colunas que o monitor produziu."""
        try:
            with open(self.complexity_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not self._complexity_header_written:
                    writer.writerow(list(comp.keys()))
                    self._complexity_header_written = True
                writer.writerow(list(comp.values()))
        except OSError as exc:
            self.logger.error("Falha ao gravar complexity_live.csv: %s", exc)

    def _salvar_relatorio_overfit(self) -> None:
        """Grava overfit_report.json: melhor epoca, inicio do overfitting e o corte (D6)."""
        try:
            with open(self.history_path, newline="", encoding="utf-8") as f:
                linhas = list(csv.DictReader(f))
            val_losses = [float(l["val_loss"]) for l in linhas if l.get("val_loss")]
        except (OSError, ValueError) as exc:
            self.logger.warning("Nao foi possivel montar o relatorio de overfitting: %s", exc)
            return
        if not val_losses:
            return

        rel = detect_overfit_onset(val_losses, patience=self.patience, delta=self.overfit_delta)
        rel["patience"] = self.patience
        rel["delta"] = self.overfit_delta
        rel["regra"] = ("As analises devem usar apenas as epocas <= melhor_epoca. "
                        "As posteriores ficam gravadas, mas marcadas como pos-overfitting "
                        "e excluidas das conclusoes (regra D6 do plan.md).")
        # O status online do monitor e provisorio (nao conhece o futuro); este relatorio,
        # calculado sobre a trajetoria completa, e quem manda. Divergencia e esperada.
        if self.monitor is not None:
            online = [h.get("status") for h in self.monitor.historico]
            n_online_of = sum(1 for st in online if st == "OVERFITTING")
            if n_online_of and not rel["confirmado"]:
                self.logger.info(
                    "O status online acusou OVERFITTING em %d epoca(s), mas a trajetoria "
                    "completa nao confirma. Vale o relatorio: nenhuma epoca sera descartada.",
                    n_online_of,
                )
            rel["status_online_overfitting_em"] = n_online_of
        destino = os.path.join(self.results_dir, "overfit_report.json")
        with open(destino, "w", encoding="utf-8") as f:
            json.dump(rel, f, indent=2, ensure_ascii=False)
        self.logger.info(
            "Overfitting | melhor_epoca=%s | inicio=%s | confirmado=%s | usaveis=%s de %s | %s",
            rel["melhor_epoca"], rel["inicio_overfit"], rel["confirmado"],
            rel["n_epocas_usaveis"], rel["n_epocas"], destino,
        )

    def _log_epoch(self, epoch: int, train_loss: float, val_loss: float, val_acc: float,
                   comp: Optional[Dict] = None, tr_eval: Optional[Dict] = None) -> None:
        linha = [epoch, train_loss, val_loss, val_acc]
        if self.eval_train:
            linha += [tr_eval["loss"], tr_eval["accuracy"]] if tr_eval else ["", ""]
        if self.monitor is not None:
            linha += ([comp["lmc"], comp["sampen"], comp["status"]] if comp else ["", "", ""])
        try:
            with open(self.history_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(linha)
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

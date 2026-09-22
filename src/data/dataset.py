"""
Datasets e DataLoaders em PyTorch puro (fase F3b do plan.md).

Substitui `monai.data`. Nenhuma dependencia do MONAI.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from ..utils.logging import get_logger


class ListaDeArquivosDataset(Dataset):
    """Dataset generico sobre uma lista de dicts {"image": caminho, "label": int}.

    Equivalente em PyTorch puro ao `monai.data.Dataset` usado antes. Usado pelo caminho
    legado que le os caminhos de um banco SQLite; o MedNIST passa por
    `src/data/mednist.py`.
    """

    def __init__(self, amostras: List[Dict], transform=None) -> None:
        self.amostras = amostras
        self.transform = transform

    def __len__(self) -> int:
        return len(self.amostras)

    def __getitem__(self, idx: int) -> Dict:
        item = self.amostras[idx]
        img = item["image"]
        if isinstance(img, (str, os.PathLike)):
            img = Image.open(img).convert("L")
        if self.transform is not None:
            img = self.transform(img)
        return {"image": img, "label": torch.tensor(int(item["label"]), dtype=torch.long)}


def _filter_missing(samples: List[Dict], logger) -> Tuple[List[Dict], List[str]]:
    filtered, missing_paths = [], []
    for sample in samples:
        caminho = sample["image"]
        if not isinstance(caminho, (str, os.PathLike)) or os.path.exists(caminho):
            filtered.append(sample)
        else:
            missing_paths.append(str(caminho))
    if missing_paths:
        for path in missing_paths[:min(10, len(missing_paths))]:
            logger.warning("Arquivo de imagem ausente, ignorado: %s", path)
    return filtered, missing_paths


def create_datasets(train_samples: List[Dict], val_samples: List[Dict], transforms,
                    cache_rate: float = 0.0, sample_log_count: int = 5, logger=None):
    """Monta os datasets a partir de listas de amostras.

    `cache_rate` e aceito por compatibilidade com chamadas antigas, mas nao tem efeito:
    o cache em memoria era um recurso do MONAI. Valores > 0 emitem aviso.
    """
    logger = logger or get_logger()
    total_input = len(train_samples) + len(val_samples)
    train_samples, faltando_train = _filter_missing(train_samples, logger)
    val_samples, faltando_val = _filter_missing(val_samples, logger)
    faltando = len(faltando_train) + len(faltando_val)
    if faltando:
        logger.warning("%d amostra(s) ignorada(s) por arquivo ausente", faltando)
        if total_input and faltando / total_input > 0.05:
            logger.warning("Mais de 5%% das amostras ausentes (%.2f%%)", faltando / total_input * 100)
    if cache_rate > 0:
        logger.warning("cache_rate=%.2f ignorado: era um recurso do MONAI, removido na F3b", cache_rate)

    for sample in train_samples[:sample_log_count]:
        logger.debug("Amostra de treino | image=%s label=%s", sample.get("image"), sample.get("label"))

    train_ds = ListaDeArquivosDataset(train_samples, transforms) if train_samples else None
    val_ds = ListaDeArquivosDataset(val_samples, transforms) if val_samples else None
    logger.info("Datasets: train=%d val=%d", len(train_samples), len(val_samples))
    return train_ds, val_ds


def create_loaders(train_ds, val_ds, test_ds=None, batch_size: int = 64, num_workers: int = 0,
                   seed: int = 0, logger=None):
    """Cria os DataLoaders dos tres conjuntos. Devolve sempre a tripla (train, val, test).

    Apenas o loader de TREINO embaralha: validacao e teste precisam de ordem fixa para
    que a metrica seja reprodutivel entre epocas e entre runs.
    """
    logger = logger or get_logger()
    kwargs = {"batch_size": batch_size, "num_workers": num_workers, "pin_memory": True}
    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(train_ds, shuffle=True, generator=g, **kwargs) if train_ds is not None else None
    val_loader = DataLoader(val_ds, shuffle=False, **kwargs) if val_ds is not None else None
    test_loader = DataLoader(test_ds, shuffle=False, **kwargs) if test_ds is not None else None
    logger.info(
        "DataLoaders criados: train=%s val=%s test=%s batch_size=%d num_workers=%d seed=%d",
        bool(train_loader), bool(val_loader), bool(test_loader), batch_size, num_workers, seed,
    )
    return train_loader, val_loader, test_loader

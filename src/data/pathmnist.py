"""
Dataset PathMNIST (MedMNIST v2/MedMNIST+) em PyTorch puro, lendo o `.npz` oficial.

Segundo conjunto de dados do projeto (decisao de 2026-10-23): repete os mesmos regimes
do MedNIST para ver se o comportamento da LMC/SampEn da camada densa se repete.

O QUE MUDA EM RELACAO AO MEDNIST
--------------------------------
  - imagens COLORIDAS (RGB) de tecido de colon corado com H&E -- nada de replicar canal
  - 9 classes (camada densa da ResNet-18 vira [9, 512]; da DenseNet-121, [9, 1024])
  - a particao treino/validacao/teste VEM PRONTA no arquivo oficial e e usada como esta:
    89.996 / 10.004 / 7.180. O teste vem de outro hospital (CRC-VAL-HE-7K), entao a
    acuracia de teste costuma ficar abaixo da de validacao -- isso e do dataset.
  - resolucao 64x64: versao oficial do MedMNIST+ (`pathmnist_64.npz`, Zenodo 10519652),
    para ficar na mesma resolucao dos runs do MedNIST.

O arquivo inteiro e carregado na memoria uma vez por processo (~1,2 GB em uint8). Os
workers do DataLoader herdam os arrays por fork, sem copiar.

O item devolvido tem o mesmo formato do MedNIST, {"image": ..., "label": ...}, para que
o Trainer, o ruido nas imagens e a subamostragem funcionem sem mudanca.
"""
from __future__ import annotations

import hashlib
import os
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

ARQUIVO_PADRAO = os.path.expanduser("~/.medmnist/pathmnist_64.npz")
SECOES = ("train", "val", "test")

# Ordem oficial dos rotulos (INFO["pathmnist"]["label"] do pacote medmnist).
CLASSES: List[str] = [
    "adipose",
    "background",
    "debris",
    "lymphocytes",
    "mucus",
    "smooth muscle",
    "normal colon mucosa",
    "cancer-associated stroma",
    "colorectal adenocarcinoma epithelium",
]

_CACHE: Dict[str, "ArquivoPathMNIST"] = {}


class ArquivoPathMNIST:
    """Le o `.npz` uma vez e guarda imagens e rotulos das tres secoes."""

    def __init__(self, caminho: str = ARQUIVO_PADRAO) -> None:
        caminho = os.path.expanduser(caminho)
        if not os.path.isfile(caminho):
            raise FileNotFoundError(
                f"PathMNIST nao encontrado: {caminho}. Baixe a versao 64x64 em "
                f"https://zenodo.org/records/10519652/files/pathmnist_64.npz"
            )
        self.caminho = caminho
        with np.load(caminho) as d:
            self.imagens = {s: d[f"{s}_images"] for s in SECOES}
            self.rotulos = {s: d[f"{s}_labels"].reshape(-1).astype(np.int64) for s in SECOES}
        for s in SECOES:
            img = self.imagens[s]
            if img.ndim != 4 or img.shape[-1] != 3 or img.dtype != np.uint8:
                raise ValueError(f"{caminho}: {s}_images com forma {img.shape} / {img.dtype}; "
                                 f"esperado [N, H, W, 3] uint8")
            if len(img) != len(self.rotulos[s]):
                raise ValueError(f"{caminho}: {s} tem {len(img)} imagens e "
                                 f"{len(self.rotulos[s])} rotulos")
        self.classes = CLASSES

    @classmethod
    def carregar(cls, caminho: str = ARQUIVO_PADRAO) -> "ArquivoPathMNIST":
        """Uma unica leitura por processo, reaproveitada pelas tres secoes."""
        chave = os.path.abspath(os.path.expanduser(caminho))
        if chave not in _CACHE:
            _CACHE[chave] = cls(caminho)
        return _CACHE[chave]

    def resumo(self) -> str:
        n = {s: len(self.rotulos[s]) for s in SECOES}
        lado = self.imagens["train"].shape[1]
        return (f"treino={n['train']} validacao={n['val']} teste={n['test']} "
                f"| {len(self.classes)} classes | {lado}x{lado} RGB | {self.caminho}")

    def sha256(self) -> str:
        """Hash do arquivo, para o config.json registrar exatamente que dado foi usado."""
        h = hashlib.sha256()
        with open(self.caminho, "rb") as f:
            for bloco in iter(lambda: f.read(1 << 20), b""):
                h.update(bloco)
        return h.hexdigest()


class PathMNISTDataset(Dataset):
    """Uma secao ("train", "val" ou "test") do PathMNIST, na particao oficial."""

    def __init__(
        self,
        secao: str,
        transform: Optional[Callable] = None,
        arquivo: Optional[ArquivoPathMNIST] = None,
        caminho: str = ARQUIVO_PADRAO,
    ) -> None:
        if secao not in SECOES:
            raise ValueError(f"secao invalida: {secao!r}. Use uma de {SECOES}.")
        self.arquivo = arquivo or ArquivoPathMNIST.carregar(caminho)
        self.secao = secao
        self.transform = transform
        self.imagens = self.arquivo.imagens[secao]
        self._rotulos = self.arquivo.rotulos[secao]
        self.classes = self.arquivo.classes

    def __len__(self) -> int:
        return len(self._rotulos)

    def __getitem__(self, idx: int) -> Dict:
        img = Image.fromarray(self.imagens[idx])  # [H, W, 3] uint8 -> RGB
        if self.transform is not None:
            img = self.transform(img)
        return {"image": img, "label": torch.tensor(int(self._rotulos[idx]), dtype=torch.long)}

    def rotulos(self) -> List[int]:
        """Rotulos na ordem do dataset, para a subamostragem estratificada."""
        return self._rotulos.tolist()

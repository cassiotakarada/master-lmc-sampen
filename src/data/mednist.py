"""
Dataset MedNIST em PyTorch puro, lendo a particao congelada (fase F3b do plan.md).

Substitui `monai.apps.MedNISTDataset`. A particao vem de `data/mednist_split.json`,
gerado uma unica vez por `scripts/congelar_particao.py` a partir do MONAI -- ver a
explicacao la sobre por que ela foi congelada em vez de reimplementada.

Vantagem sobre a versao anterior: a divisao treino/validacao/teste deixa de depender da
implementacao interna de uma biblioteca e passa a ser um dado versionado do projeto.
Qualquer pessoa reproduz exatamente a mesma divisao.

O item devolvido mantem o formato de dicionario {"image": ..., "label": ...} usado pelo
resto do codigo, para que o Trainer nao precise mudar.
"""
from __future__ import annotations

import json
import os
from typing import Callable, Dict, List, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset

ARQUIVO_PARTICAO_PADRAO = "data/mednist_split.json"
SECOES = ("train", "val", "test")


class ParticaoMedNIST:
    """Le e valida `mednist_split.json` uma vez, reaproveitado pelas tres secoes."""

    def __init__(self, caminho: str = ARQUIVO_PARTICAO_PADRAO) -> None:
        if not os.path.isfile(caminho):
            raise FileNotFoundError(
                f"Particao nao encontrada: {caminho}. "
                f"Gere-a com: python scripts/congelar_particao.py"
            )
        with open(caminho, encoding="utf-8") as f:
            self.dados = json.load(f)
        self.caminho = caminho
        self.classes: List[str] = self.dados["classes"]
        self.sha256: str = self.dados.get("sha256", "")

    def itens(self, secao: str) -> List[Dict]:
        if secao not in SECOES:
            raise ValueError(f"secao invalida: {secao!r}. Use uma de {SECOES}.")
        return self.dados["splits"][secao]

    def resumo(self) -> str:
        d = self.dados
        return (f"treino={d['n_train']} validacao={d['n_val']} teste={d['n_test']} "
                f"| {len(self.classes)} classes | sha256={self.sha256[:12]}")


class MedNISTDataset(Dataset):
    """MedNIST a partir da particao congelada.

    Parametros
    ----------
    secao     : "train", "val" ou "test"
    transform : callable aplicado a imagem PIL (ver src/data/transforms.py)
    particao  : ParticaoMedNIST ja carregada, ou None para carregar do caminho padrao
    raiz      : prefixo opcional para os caminhos das imagens (util se o dataset mudar de lugar)
    """

    def __init__(
        self,
        secao: str,
        transform: Optional[Callable] = None,
        particao: Optional[ParticaoMedNIST] = None,
        arquivo_particao: str = ARQUIVO_PARTICAO_PADRAO,
        raiz: str = "",
    ) -> None:
        self.particao = particao or ParticaoMedNIST(arquivo_particao)
        self.secao = secao
        self.transform = transform
        self.raiz = raiz
        self.data: List[Dict] = self.particao.itens(secao)
        self.classes = self.particao.classes

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]
        caminho = os.path.join(self.raiz, item["image"]) if self.raiz else item["image"]
        # MedNIST e escala de cinza; "L" garante 1 canal mesmo se o arquivo vier RGB
        img = Image.open(caminho).convert("L")
        if self.transform is not None:
            img = self.transform(img)
        return {"image": img, "label": torch.tensor(item["label"], dtype=torch.long)}

    def rotulos(self) -> List[int]:
        """Rotulos na ordem do dataset, sem abrir nenhuma imagem.

        Usado pela subamostragem estratificada (`train_fraction`), que precisa das
        classes mas nao dos pixels.
        """
        return [int(d["label"]) for d in self.data]

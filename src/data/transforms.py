"""
Transformacoes em torchvision puro (fase F3b do plan.md).

Substitui as transformacoes do MONAI. Espelha o pipeline ja usado no repositorio da
dissertacao (`thesis-complexity-nn/src/training.py`), para que os dois projetos
processem as imagens do mesmo jeito.

DECISAO: replicar o canal cinza para 3 canais
---------------------------------------------
O MedNIST e escala de cinza (1 canal), mas as redes do torchvision esperam 3. Ha duas
saidas: (a) trocar a primeira convolucao por uma de 1 canal, ou (b) replicar o canal
tres vezes. Adotamos (b) porque e o que a dissertacao ja faz -- manter os dois
pipelines iguais vale mais do que a pequena economia de (a). A camada densa analisada
(`fc`, 6x512) e identica nos dois casos.
"""
from __future__ import annotations

from typing import Sequence

from torchvision import transforms


def _para_rgb(x):
    """Replica o unico canal em tres. Entra [1,H,W], sai [3,H,W]."""
    return x.repeat(3, 1, 1) if x.shape[0] == 1 else x


def build_transforms(size: Sequence[int] = (224, 224), treino: bool = False):
    """Pipeline de transformacoes.

    treino=True acrescenta espelhamento horizontal aleatorio (aumento de dados). A
    validacao e o teste NUNCA usam aumento: a regua tem de ser fixa.
    """
    etapas = []
    if treino:
        etapas.append(transforms.RandomHorizontalFlip())
    etapas += [
        transforms.Resize(tuple(size)),
        transforms.ToTensor(),          # PIL [0,255] -> tensor [0,1], shape [1,H,W]
        transforms.Lambda(_para_rgb),   # [1,H,W] -> [3,H,W]
    ]
    return transforms.Compose(etapas)

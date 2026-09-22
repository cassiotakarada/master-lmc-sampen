"""
Construcao do modelo em torchvision puro (fase F3b do plan.md).

Substitui `monai.networks.nets`. Espelha `thesis-complexity-nn/src/training.py`, para
que os dois projetos usem exatamente as mesmas arquiteturas.

A CAMADA DENSA ANALISADA
------------------------
  resnet18     -> `fc`         = Linear(512, num_classes)   -> matriz [num_classes, 512]
  densenet121  -> `classifier` = Linear(1024, num_classes)  -> matriz [num_classes, 1024]

Com 6 classes, a ResNet-18 da a matriz [6, 512] = 3.072 pesos que o projeto analisa --
a mesma forma da versao anterior com MONAI.

ATENCAO: a ResNet-18 do torchvision NAO e numericamente identica a do MONAI (que deriva
de uma ResNet 3D medica, com tronco diferente). Runs novos nao sao comparaveis com os
anteriores a F3b; formam uma nova linha de base.
"""
from __future__ import annotations

import torch.nn as nn
from torchvision import models

ARQUITETURAS = ("resnet18", "resnet34", "densenet121")


def build_model(model_name: str, in_channels: int = 3, num_classes: int = 6) -> nn.Module:
    """Cria a rede com a cabeca de classificacao ajustada ao numero de classes.

    `in_channels` e aceito por compatibilidade com o config, mas o pipeline replica o
    canal cinza para 3 (ver src/data/transforms.py), entao o valor efetivo e sempre 3.
    Passar outro valor emite erro, para nao criar uma divergencia silenciosa.
    """
    nome = model_name.lower()
    if in_channels != 3:
        raise ValueError(
            f"in_channels={in_channels}: o pipeline replica o canal cinza para 3 canais "
            f"(ver src/data/transforms.py). Use in_channels=3."
        )

    if nome == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif nome == "resnet34":
        model = models.resnet34(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif nome == "densenet121":
        model = models.densenet121(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    else:
        raise ValueError(f"Arquitetura nao suportada: {model_name!r}. Use uma de {ARQUITETURAS}.")
    return model

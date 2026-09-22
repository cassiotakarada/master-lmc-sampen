import json
import tempfile

import pytest
import torch.nn as nn

from src.training.model import ARQUITETURAS, build_model
from src.utils.param_types import save_param_types


def test_param_types_saved():
    model = build_model("densenet121", in_channels=3, num_classes=2)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/param_types.json"
        save_param_types(model, path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    assert isinstance(data, dict)
    assert len(data) > 0
    valid_categories = {"Convolutional", "Linear", "Embedding", "BatchNorm", "Activation", "Pooling", "Flatten", "Other"}
    assert set(data.values()).issubset(valid_categories)


def test_in_channels_diferente_de_3_e_rejeitado():
    """Desde a F3b o pipeline replica o cinza para 3 canais; aceitar 1 em silencio
    criaria divergencia entre o que o config diz e o que a rede recebe."""
    with pytest.raises(ValueError, match="in_channels"):
        build_model("resnet18", in_channels=1, num_classes=6)


def test_arquitetura_desconhecida_e_rejeitada():
    with pytest.raises(ValueError, match="nao suportada"):
        build_model("vgg16", in_channels=3, num_classes=6)


@pytest.mark.parametrize("arq,esperado", [
    ("resnet18", (6, 512)),
    ("resnet34", (6, 512)),
    ("densenet121", (6, 1024)),
])
def test_forma_da_camada_densa(arq, esperado):
    """A camada densa e o objeto de estudo do projeto: sua forma nao pode mudar sem aviso.

    A ResNet-18 com 6 classes da [6, 512] = 3072 pesos -- a mesma forma de antes da
    remocao do MONAI, o que mantem toda a analise valida.
    """
    model = build_model(arq, in_channels=3, num_classes=6)
    densas = [m for m in model.modules() if isinstance(m, nn.Linear)]
    assert tuple(densas[-1].weight.shape) == esperado


def test_arquiteturas_declaradas():
    assert ARQUITETURAS == ("resnet18", "resnet34", "densenet121")

"""
Ruido de rotulo: como induzir overfitting de verdade (fase F6 do plan.md).

POR QUE NAO REDUZIR O CONJUNTO DE TREINO
----------------------------------------
A primeira ideia do plano era induzir overfitting treinando com poucos dados
(`train_fraction` pequeno). A F3 mostrou que isso produz um FALSO POSITIVO convincente:
com poucos passos, as estatisticas moveis do BatchNorm nao convergem e o modelo desaba
em eval() -- inclusive sobre o proprio conjunto de treino. Parece overfitting
catastrofico, mas o modelo acerta 94,5% da validacao quando o BN usa estatisticas do
lote. Media-se um artefato de normalizacao e chamar-se-ia de overfitting.

Ruido de rotulo nao tem esse problema: mantem as 47.164 amostras, entao o BatchNorm
continua saudavel, e forca memorizacao genuina -- nao existe regra visual que explique
rotulos sorteados ao acaso, so da para decorar.

E a tecnica padrao da literatura de generalizacao (Zhang et al., 2017,
"Understanding deep learning requires rethinking generalization").

SOMENTE O TREINO E CORROMPIDO. Validacao e teste ficam intactos -- corromper a regua
junto com o experimento nao mediria nada.
"""
from __future__ import annotations

import random
from typing import Dict, List, Sequence

from torch.utils.data import Dataset


def corromper_rotulos(
    rotulos: Sequence[int],
    fracao: float,
    n_classes: int,
    seed: int,
) -> tuple[List[int], Dict]:
    """Troca o rotulo de `fracao` das amostras por um sorteado uniformemente.

    Convencao adotada (padrao de Zhang et al., 2017): o novo rotulo e sorteado entre
    TODAS as classes, inclusive a verdadeira. Com `n_classes` classes, uma fracao f
    sorteada resulta em f*(n_classes-1)/n_classes de rotulos efetivamente ERRADOS --
    com 30% e 6 classes, 25% ficam de fato errados. O dicionario devolvido reporta o
    numero efetivo, para nao confundir "fracao sorteada" com "fracao errada".

    Determinista: mesma (rotulos, fracao, seed) devolve sempre o mesmo resultado.
    """
    if not 0.0 <= fracao <= 1.0:
        raise ValueError(f"fracao deve estar em [0, 1], recebi {fracao}")
    originais = [int(r) for r in rotulos]
    n = len(originais)
    if fracao == 0.0 or n == 0:
        return originais, {"fracao_sorteada": 0.0, "n_sorteados": 0,
                           "n_efetivamente_errados": 0, "fracao_efetiva": 0.0}

    rng = random.Random(seed)
    n_sortear = round(n * fracao)
    alvos = rng.sample(range(n), n_sortear)

    novos = list(originais)
    for i in alvos:
        novos[i] = rng.randrange(n_classes)

    errados = sum(1 for i in range(n) if novos[i] != originais[i])
    return novos, {
        "fracao_sorteada": fracao,
        "n_sorteados": n_sortear,
        "n_efetivamente_errados": errados,
        "fracao_efetiva": errados / n,
        "n_classes": n_classes,
        "seed": seed,
    }


class RuidoDeRotulo(Dataset):
    """Envolve um dataset e devolve rotulos corrompidos, sem tocar nas imagens.

    O dataset original continua intacto: so o rotulo devolvido muda.
    """

    def __init__(self, base: Dataset, fracao: float, n_classes: int, seed: int) -> None:
        if not hasattr(base, "rotulos"):
            raise TypeError("o dataset base precisa do metodo rotulos()")
        self.base = base
        self.rotulos_corrompidos, self.relatorio = corromper_rotulos(
            base.rotulos(), fracao, n_classes, seed
        )

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Dict:
        item = self.base[idx]
        import torch
        item["label"] = torch.tensor(self.rotulos_corrompidos[idx], dtype=torch.long)
        return item

    def rotulos(self) -> List[int]:
        return list(self.rotulos_corrompidos)

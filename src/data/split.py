"""
Particionamento e subamostragem reprodutiveis dos dados (fase F3 do plan.md).

DISCIPLINA DOS TRES CONJUNTOS
-----------------------------
  treino     -- unico que entra no backward(); ajusta os pesos
  validacao  -- escolhe a epoca e detecta overfitting; NUNCA entra no gradiente
  teste      -- mede o desempenho final; olhado UMA UNICA VEZ, no fim

Analogia: treino = lista de exercicios, validacao = simulado, teste = a prova.
Quem estuda pela prova nao sabe se aprendeu.

SEMENTE DA PARTICAO vs SEMENTE DO TREINO
----------------------------------------
Sao duas coisas diferentes e precisam ficar separadas:

  - `data_seed` decide QUEM cai em treino/validacao/teste. Fica FIXA entre todos
    os runs. Se ela variasse junto com a semente do treino, cada run veria uma
    particao diferente, os runs deixariam de ser comparaveis e uma amostra que foi
    teste num run seria treino no outro -- vazamento entre runs.
  - `seed` (do TrainConfig) decide a inicializacao dos pesos e a ordem dos lotes.
    Essa sim varia entre repeticoes, que e o ponto de repetir.
"""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List, Sequence


def stratified_subset_indices(
    labels: Sequence[int],
    fraction: float,
    seed: int,
    min_por_classe: int = 1,
) -> List[int]:
    """
    Sorteia uma fracao dos indices preservando a proporcao entre classes.

    Usado no run de "overfitting induzido" da F6: treinar com poucos dados e muita
    capacidade e a forma mais limpa de produzir overfitting inequivoco. Sortear sem
    estratificar poderia zerar uma classe inteira e trocar o fenomeno estudado
    (overfitting) por outro (desbalanceamento).

    Determinista: a mesma (labels, fraction, seed) devolve sempre os mesmos indices,
    em ordem crescente.

    Parametros
    ----------
    labels          : rotulo de cada amostra, na ordem do dataset
    fraction        : fracao a manter, em (0, 1]. 1.0 devolve todos os indices.
    seed            : semente do sorteio
    min_por_classe  : minimo de amostras por classe, para nenhuma classe sumir

    Levanta ValueError se fraction estiver fora de (0, 1].
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction deve estar em (0, 1], recebi {fraction}")
    n = len(labels)
    if fraction == 1.0:
        return list(range(n))

    por_classe: Dict[int, List[int]] = defaultdict(list)
    for idx, lab in enumerate(labels):
        por_classe[int(lab)].append(idx)

    rng = random.Random(seed)
    escolhidos: List[int] = []
    for classe in sorted(por_classe):
        indices = por_classe[classe]
        quantos = max(min_por_classe, round(len(indices) * fraction))
        quantos = min(quantos, len(indices))
        escolhidos.extend(rng.sample(indices, quantos))

    return sorted(escolhidos)


def resumo_particao(n_train: int, n_val: int, n_test: int) -> str:
    """String de uma linha para o log e para o config.json."""
    total = n_train + n_val + n_test
    if total == 0:
        return "particao vazia"
    return (
        f"treino={n_train} ({n_train/total:.1%}) | "
        f"validacao={n_val} ({n_val/total:.1%}) | "
        f"teste={n_test} ({n_test/total:.1%}) | total={total}"
    )

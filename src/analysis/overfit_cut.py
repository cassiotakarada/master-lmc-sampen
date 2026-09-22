"""
Aplicacao da regra D6: os dados apos o overfitting nao devem ser utilizados.

Fonte unica desta regra para TODOS os scripts de analise. O criterio em si vive em
`src.training.overfit`; aqui esta o que os consumidores precisam: carregar o relatorio
de um run, marcar as epocas descartadas num DataFrame e sombrear a regiao no grafico.

O QUE "DESCARTAR" SIGNIFICA AQUI
--------------------------------
Nao significa apagar. As epocas posteriores continuam gravadas em disco e continuam
aparecendo nos graficos -- sombreadas e com a legenda "descartado". O que elas NAO fazem
e entrar em nenhuma conclusao, media, correlacao ou tabela de resultado.

Mostrar a regiao descartada e melhor do que cortar o eixo: o leitor ve o que aconteceu
depois e confere por si mesmo que a linha de corte esta no lugar certo.
"""
from __future__ import annotations

import json
import os
from typing import Dict, Optional

import pandas as pd

from ..training.overfit import detect_degradacao_acuracia, detect_overfit_onset

NOME_RELATORIO = "overfit_report.json"
COR_DESCARTE = "#e34948"   # vermelho da paleta categorica


def carregar_relatorio(run_dir: str, patience: int = 3, delta: float = 0.0,
                       criterio: str = "val_loss") -> Optional[Dict]:
    """Le `overfit_report.json` do run; se nao existir, recalcula do `history.csv`.

    O fallback importa: todos os runs anteriores a F4 foram treinados sem o relatorio,
    e nao faz sentido exigir re-treino so para aplicar o corte a eles.

    Devolve None se nao houver nem relatorio nem history.csv utilizavel.

    Se `criterio="val_accuracy"`, o corte e ancorado na queda sustentada da acuracia em
    vez do minimo do val_loss -- mais robusto quando a metrica satura (ver
    `detect_degradacao_acuracia`). Nesse modo o relatorio e sempre recalculado.
    """
    if criterio == "val_accuracy":
        hist = os.path.join(run_dir, "history.csv")
        if not os.path.isfile(hist):
            return None
        try:
            va = pd.read_csv(hist)["val_accuracy"].dropna().astype(float).tolist()
        except (KeyError, ValueError):
            return None
        if not va:
            return None
        rel = detect_degradacao_acuracia(va, patience=patience)
        rel["_origem"] = "recalculado de history.csv (criterio val_accuracy)"
        return rel

    caminho = os.path.join(run_dir, NOME_RELATORIO)
    if os.path.isfile(caminho):
        with open(caminho, encoding="utf-8") as f:
            rel = json.load(f)
        rel["_origem"] = "overfit_report.json"
        return rel

    hist = os.path.join(run_dir, "history.csv")
    if not os.path.isfile(hist):
        return None
    try:
        df = pd.read_csv(hist)
        val = df["val_loss"].dropna().astype(float).tolist()
    except (KeyError, ValueError):
        return None
    if not val:
        return None

    rel = detect_overfit_onset(val, patience=patience, delta=delta)
    rel["patience"] = patience
    rel["delta"] = delta
    rel["_origem"] = "recalculado de history.csv"
    return rel


def marcar_pos_overfit(df: pd.DataFrame, relatorio: Optional[Dict],
                       coluna_epoca: str = "epoch") -> pd.DataFrame:
    """Acrescenta a coluna booleana `pos_overfit` ao DataFrame (copia, nao muta).

    True = epoca posterior a melhor_epoca num run com overfitting CONFIRMADO. Se o
    overfitting nao foi confirmado, nada e descartado: nao ha regiao pos-overfitting.
    """
    out = df.copy()
    if not relatorio or not relatorio.get("confirmado"):
        out["pos_overfit"] = False
        return out
    melhor = int(relatorio["melhor_epoca"])
    out["pos_overfit"] = out[coluna_epoca].astype(int) > melhor
    return out


def apenas_usaveis(df: pd.DataFrame, relatorio: Optional[Dict],
                   coluna_epoca: str = "epoch") -> pd.DataFrame:
    """Devolve so as epocas que podem entrar em conclusoes (regra D6)."""
    marcado = marcar_pos_overfit(df, relatorio, coluna_epoca)
    return marcado[~marcado["pos_overfit"]].drop(columns=["pos_overfit"])


def sombrear_regiao_descartada(ax, relatorio: Optional[Dict], com_legenda: bool = True) -> bool:
    """Sombreia no eixo a regiao pos-overfitting e marca a melhor epoca.

    Devolve True se sombreou. Nao faz nada se o overfitting nao foi confirmado --
    sombrear um run que nunca entrou em overfitting seria mentir sobre o dado.
    """
    if not relatorio or not relatorio.get("confirmado"):
        return False
    melhor = float(relatorio["melhor_epoca"])
    x_max = ax.get_xlim()[1]
    ax.axvspan(melhor, x_max, color=COR_DESCARTE, alpha=0.08, zorder=0,
               label="descartado (pós-overfitting)" if com_legenda else None)
    ax.axvline(melhor, color=COR_DESCARTE, linewidth=1.4, linestyle="--", zorder=1,
               label=f"melhor época ({int(melhor)})" if com_legenda else None)
    return True


def resumo_texto(relatorio: Optional[Dict]) -> str:
    """Uma linha para titulo de figura, log ou tabela."""
    if not relatorio:
        return "sem relatório de overfitting"
    if not relatorio.get("confirmado"):
        return (f"sem overfitting confirmado (melhor época {relatorio.get('melhor_epoca')}"
                f" de {relatorio.get('n_epocas')}) — nenhuma época descartada")
    return (f"melhor época {relatorio['melhor_epoca']}; "
            f"{relatorio['n_epocas_descartadas']} de {relatorio['n_epocas']} épocas descartadas")

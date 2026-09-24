"""
Analise da fase F6: os indicadores antecipam o overfitting? (plan.md)

Le o `complexity_live.csv` produzido pelo monitor durante o treino -- LMC, SampEn 1D e
SampEn2D por epoca, ja calculadas sobre TODOS os 3.072 pesos da camada densa -- e o
cruza com a trajetoria de val_loss.

A PERGUNTA QUE ESTE SCRIPT RESPONDE
-----------------------------------
A LMC (ou a SampEn) muda de regime ANTES de o val_loss comecar a piorar? Se sim, de
quantas epocas e essa antecedencia? Se nao, isso precisa ser reportado como tal.

Convencao de sinal: antecedencia POSITIVA = o indicador virou ANTES do val_loss (util);
NEGATIVA = virou depois (inutil como aviso).

Regra D6: as epocas posteriores a melhor continuam nos graficos, sombreadas, mas nao
entram em nenhuma conclusao.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis.overfit_cut import carregar_relatorio, resumo_texto, sombrear_regiao_descartada

# paleta categorica validada (slots 1, 2, 3)
COR = {"lmc": "#2a78d6", "sampen": "#eb6834", "sampen2d": "#1baf7a",
       "val": "#0b0b0b", "train": "#e34948"}
SURFACE, TEXT_1, TEXT_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2de"

INDICADORES = [("lmc", "LMC"), ("sampen", "SampEn 1D"), ("sampen2d", "SampEn2D")]


def _eixo(ax, xlabel, ylabel, titulo):
    ax.set_facecolor(SURFACE)
    ax.set_xlabel(xlabel, color=TEXT_2, fontsize=9)
    ax.set_ylabel(ylabel, color=TEXT_2, fontsize=9)
    ax.set_title(titulo, color=TEXT_1, fontsize=10.5, loc="left", pad=8)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    ax.tick_params(colors=TEXT_2, labelsize=8.5)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(GRID)


# ---------------------------------------------------------------------------
# Deteccao do sinal e marcos de degradacao
# ---------------------------------------------------------------------------
# CALIBRACAO (protocolo do plan.md): estes parametros sao ajustados olhando os runs
# NORMAIS -- onde nao ha overfitting e o indicador NAO deve disparar -- e so depois
# aplicados ao run de ruido. Os normais funcionam como controle negativo.
# CALIBRACAO (protocolo do plan.md): estes parametros sao ajustados olhando os runs
# NORMAIS -- onde nao ha overfitting e o indicador NAO deve disparar -- e so depois
# aplicados ao run de ruido. Os normais funcionam como controle negativo.
BASE_EPOCAS = 5      # epocas iniciais que definem o plato de referencia
BASE_K = 4.0         # quantos desvios do plato contam como afastamento
BASE_P = 3           # epocas consecutivas fora da faixa para confirmar

# O detector mora em src/training/overfit.py desde 2026-09-22 -- o status online (F4) e
# a analise offline (F6) tem de usar exatamente o mesmo criterio, entao ele nao pode
# existir em duas copias. Reexportado aqui pelo nome de sempre, para nao quebrar quem
# importa deste modulo.
from src.training.overfit import epoca_de_afastamento as _afastamento


def epoca_de_afastamento(serie, base_epocas: int = BASE_EPOCAS, k: float = BASE_K,
                         p: int = BASE_P):
    """Ver `src.training.overfit.epoca_de_afastamento` -- fonte unica do criterio."""
    return _afastamento(serie, base_epocas, k, p)


def marcos_de_degradacao(df: pd.DataFrame, queda_pp: float = 0.01, p: int = 3) -> dict:
    """Quando o desempenho de fato comeca a piorar -- tres ancoras, nao uma.

    O minimo do val_loss e a ancora padrao de early stopping, mas e fragil quando a
    metrica esta saturada: no run de ruido ele cai na epoca 4, com a acuracia ainda em
    99,75%. Por isso reportamos tambem duas ancoras baseadas em acuracia, mais robustas.
    """
    m = {}
    if "val_loss" in df:
        m["min_val_loss"] = int(df.loc[df["val_loss"].idxmin(), "epoch"])
    if "val_accuracy" in df:
        va = df["val_accuracy"].to_numpy()
        pico = float(np.nanmax(va))
        # Queda SUSTENTADA: `p` epocas consecutivas abaixo do pico. Sem essa exigencia,
        # uma unica epoca ruim dispararia a ancora -- e nos runs normais, onde a acuracia
        # oscila em torno de 99,9%, isso acontecia ja na epoca 2.
        m["val_acc_cai"] = None
        for i in range(len(va) - p + 1):
            if np.all(va[i:i + p] < pico - queda_pp):
                m["val_acc_cai"] = int(df["epoch"].iloc[i])
                break
        m["val_acc_pico"] = int(df.loc[df["val_accuracy"].idxmax(), "epoch"])
    if "train_eval_accuracy" in df and "val_accuracy" in df:
        cruz = df[df["train_eval_accuracy"] > df["val_accuracy"]]
        m["treino_ultrapassa_val"] = int(cruz["epoch"].iloc[0]) if len(cruz) else None
    return m


def analisar_run(run_dir: str) -> dict | None:
    caminho = os.path.join(run_dir, "complexity_live.csv")
    if not os.path.isfile(caminho):
        print(f"[skip] sem complexity_live.csv: {run_dir}")
        return None
    df = pd.read_csv(caminho).sort_values("epoch").reset_index(drop=True)
    # history.csv traz val_accuracy e train_eval_accuracy, que o monitor nao registra
    hist_path = os.path.join(run_dir, "history.csv")
    if os.path.isfile(hist_path):
        hist = pd.read_csv(hist_path)
        extras = [c for c in ("val_accuracy", "train_eval_accuracy", "train_eval_loss",
                              "train_loss") if c in hist.columns]
        df = df.merge(hist[["epoch"] + extras], on="epoch", how="left", suffixes=("", "_h"))
    # Criterio por ACURACIA: o val_loss satura neste problema e marcaria a epoca 4,
    # com a acuracia ainda em 99,75%. Ver detect_degradacao_acuracia.
    rel = carregar_relatorio(run_dir, criterio="val_accuracy")
    nome = os.path.basename(run_dir.rstrip("/"))

    marcos = marcos_de_degradacao(df)
    # Ancora principal: queda de 1 ponto percentual na acuracia de validacao.
    # Mais robusta que o minimo do val_loss quando a metrica esta saturada.
    ancora = marcos.get("val_acc_cai")

    linha = {"run": nome, "n_epocas": len(df),
             "overfit_confirmado": bool(rel.get("confirmado")) if rel else False,
             "min_val_loss": marcos.get("min_val_loss"),
             "val_acc_cai": marcos.get("val_acc_cai"),
             "treino_passa_val": marcos.get("treino_ultrapassa_val")}

    for chave, rotulo in INDICADORES:
        if chave not in df.columns:
            continue
        serie = df[chave].to_numpy()
        sinal = epoca_de_afastamento(serie)
        linha[f"{chave}_sinal"] = sinal
        # POSITIVO = o indicador disparou ANTES de o desempenho piorar (aviso util)
        linha[f"{chave}_antec"] = (ancora - sinal) if (ancora and sinal) else None
        # AMPLITUDE RELATIVA: e ela que separa um run com overfitting de um normal.
        # Amplitude grande NAO basta -- a LMC varia ~110-125% em TODOS os runs, com ou
        # sem overfitting, entao amplitude alta sozinha nao indica overfitting. O que
        # vale e o CONTRASTE entre o run com ruido e os controles.
        finita = serie[np.isfinite(serie)]
        linha[f"{chave}_amplitude"] = (
            (finita.max() - finita.min()) / abs(finita.mean()) if len(finita) and finita.mean() else None
        )
    return {"linha": linha, "df": df, "rel": rel, "nome": nome, "marcos": marcos}


def figura_run(res: dict, out: str) -> None:
    """Paineis lado a lado -- NUNCA dois eixos y (ver plan.md, nota da F2)."""
    df, rel, nome = res["df"], res["rel"], res["nome"]
    tem_train = "train_eval_accuracy" in df.columns

    n_paineis = 1 + len(INDICADORES)
    fig, axes = plt.subplots(1, n_paineis, figsize=(4.0 * n_paineis, 4.4), facecolor=SURFACE)
    x = df["epoch"].to_numpy()

    ax = axes[0]
    if "val_accuracy" in df.columns:
        ax.plot(x, df["val_accuracy"], color=COR["val"], linewidth=1.8, marker="o",
                markersize=3.5, markeredgecolor=SURFACE, markeredgewidth=0.6,
                label="acurácia de validação")
        if "train_eval_accuracy" in df.columns:
            ax.plot(x, df["train_eval_accuracy"], color=COR["train"], linewidth=1.8,
                    marker="s", markersize=3.5, markeredgecolor=SURFACE,
                    markeredgewidth=0.6, label="acurácia de treino (eval)")
        anc = res["marcos"].get("val_acc_cai")
        if anc:
            ax.axvline(anc, color=COR["val"], linestyle="--", linewidth=1.2, alpha=0.7,
                       label=f"val_acc cai (época {anc})")
        _eixo(ax, "Época", "Acurácia", "A. Desempenho (o que queremos antecipar)")
    else:
        ax.plot(x, df["val_loss"], color=COR["val"], linewidth=1.8, marker="o", markersize=3.5,
                markeredgecolor=SURFACE, markeredgewidth=0.6, label="val_loss")
        _eixo(ax, "Época", "Perda de validação", "A. Desempenho (o que queremos antecipar)")
    sombrear_regiao_descartada(ax, rel)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=TEXT_2)

    for i, (chave, rotulo) in enumerate(INDICADORES, start=1):
        ax = axes[i]
        if chave not in df.columns:
            ax.set_visible(False)
            continue
        ax.plot(x, df[chave], color=COR[chave], linewidth=1.8, marker="o", markersize=3.5,
                markeredgecolor=SURFACE, markeredgewidth=0.6, label=rotulo)
        sinal = res["linha"].get(f"{chave}_sinal")
        if sinal:
            ax.axvline(sinal, color=COR[chave], linestyle=":", linewidth=1.6,
                       label=f"sinal (época {sinal})")
        anc = res["marcos"].get("val_acc_cai")
        if anc:
            ax.axvline(anc, color=COR["val"], linestyle="--", linewidth=1.2,
                       alpha=0.7, label=f"val_acc cai (época {anc})")
        _eixo(ax, "Época", rotulo, f"{'BCD'[i-1]}. {rotulo}")
        sombrear_regiao_descartada(ax, rel, com_legenda=False)
        ax.legend(frameon=False, fontsize=8.5, labelcolor=TEXT_2)

    sub = resumo_texto(rel)
    if tem_train:
        g = df["train_eval_accuracy"].iloc[-1] - df["val_accuracy"].iloc[-1]
        sub += f" | acurácia final treino−validação = {g:+.3f}"
    fig.suptitle(f"Indicadores de complexidade × desempenho — {nome}",
                 color=TEXT_1, fontsize=12.5, x=0.006, ha="left", y=0.985, weight="bold")
    fig.text(0.006, 0.935, sub, color=TEXT_2, fontsize=8.5, ha="left", va="top")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def figura_memorizacao(res: dict, out: str) -> bool:
    """Para o run de ruido: treino sobe e validacao cai? (checagem anti-BatchNorm)"""
    df = res["df"]
    if "train_eval_accuracy" not in df.columns:
        return False
    fig, ax = plt.subplots(figsize=(7.5, 4.6), facecolor=SURFACE)
    x = df["epoch"].to_numpy()
    ax.plot(x, df["train_eval_accuracy"], color=COR["train"], linewidth=2, marker="o",
            markersize=4, markeredgecolor=SURFACE, label="treino (em eval)")
    ax.plot(x, df["val_accuracy"], color=COR["val"], linewidth=2, marker="s",
            markersize=4, markeredgecolor=SURFACE, label="validação")
    _eixo(ax, "Época", "Acurácia",
          "Memorização real ou artefato? Treino ALTO + validação BAIXA = overfitting")
    sombrear_regiao_descartada(ax, res["rel"])
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT_2)
    fig.text(0.008, 0.955,
             "Se AMBAS caírem, não é overfitting: é artefato de BatchNorm (ver plan.md, achado da F3).",
             color=TEXT_2, fontsize=8.5, ha="left", va="top")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return True


def main() -> int:
    p = argparse.ArgumentParser(description="Analise da F6")
    p.add_argument("--run_dirs", nargs="+", default=None)
    p.add_argument("--root", default="monai_weights")
    p.add_argument("--output_dir", default="monai_weights/f6_analise")
    args = p.parse_args()

    alvos = args.run_dirs or sorted(
        os.path.join(args.root, d) for d in os.listdir(args.root)
        if d.startswith("f6_") and os.path.isfile(os.path.join(args.root, d, "complexity_live.csv"))
    )
    if not alvos:
        print("Nenhum run f6_* com complexity_live.csv encontrado.")
        return 1

    os.makedirs(args.output_dir, exist_ok=True)
    linhas = []
    for rd in alvos:
        res = analisar_run(rd)
        if not res:
            continue
        linhas.append(res["linha"])
        figura_run(res, os.path.join(args.output_dir, f"indicadores_{res['nome']}.png"))
        if figura_memorizacao(res, os.path.join(args.output_dir, f"memorizacao_{res['nome']}.png")):
            print(f"  [{res['nome']}] figura de memorizacao gerada")
        print(f"  [{res['nome']}] figura de indicadores gerada")

    tab = pd.DataFrame(linhas)
    tab.to_csv(os.path.join(args.output_dir, "f6_resumo.csv"), index=False)

    print("\n" + "=" * 100)
    print("RESULTADO CENTRAL DA F6 — os indicadores antecipam a piora da validacao?")
    print("=" * 100)
    print(tab.to_string(index=False))
    # --- contraste entre o run com overfitting e os controles: o resultado central ---
    com = tab[tab["run"].str.contains("ruido")]
    ctrl = tab[~tab["run"].str.contains("ruido")]
    if len(com) and len(ctrl):
        print("\n" + "=" * 100)
        print("PODER DE DISCRIMINACAO — amplitude relativa no run com overfitting vs controles")
        print("=" * 100)
        print(f"{'indicador':<14} {'com overfitting':>17} {'controles (media)':>19} {'razao':>9}  veredito")
        for chave, rotulo in INDICADORES:
            col = f"{chave}_amplitude"
            if col not in tab.columns:
                continue
            a = float(com[col].mean())
            b = float(ctrl[col].mean())
            razao = a / b if b else float("inf")
            v = "DISCRIMINA" if razao >= 3 else ("fraco" if razao >= 1.5 else "NAO DISCRIMINA")
            print(f"{rotulo:<14} {a:>16.1%} {b:>18.1%} {razao:>8.1f}x  {v}")
        print("\nAmplitude grande NAO e o criterio: o que importa e a RAZAO entre o run com")
        print("overfitting e os controles. Um indicador que varia muito nos dois nao avisa nada.")

    print("\nancora = epoca em que a acuracia de validacao cai 1 ponto percentual, de forma sustentada.")
    print("antec POSITIVA = o indicador disparou ANTES da queda (aviso util).")
    print("antec NEGATIVA = disparou depois. Vazio = nunca se afastou do plato.")
    print("Nos runs NORMAIS (controle), o esperado e NAO disparar.")

    with open(os.path.join(args.output_dir, "f6_resumo.json"), "w", encoding="utf-8") as f:
        json.dump(linhas, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nSalvo em {args.output_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())

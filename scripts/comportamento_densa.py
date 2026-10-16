"""
Comportamento da LMC e da SampEn na camada densa, por regime de treino.

Escopo DESCRITIVO (decisao de 2026-10-01): nao ha detector, limiar nem "status". O
script so le o `complexity_live.csv` de cada run -- LMC, SampEn 1D e SampEn2D ja
calculadas epoca a epoca sobre os 3.072 pesos de `fc.weight` -- e descreve como cada
estatistica se comporta em tres regimes:

  controle       100% dos dados, rotulos corretos            (f6_ctrlpar_seed*)
  ruido          100% dos dados, 15/30/50% dos rotulos trocados (f6_ruido*_seed*)
  subamostragem  10/2/1% dos dados, rotulos corretos          (f6_frac*_seed*)

Alem do eixo de epocas, usa o eixo de PASSOS DE OTIMIZACAO: com 2% dos dados uma
epoca tem 8 passos, com 100% tem 369. Comparar por epoca mistura "quanto dado" com
"quanto treino".
"""
from __future__ import annotations

import argparse
import glob
import json
import math
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
from scipy.stats import mannwhitneyu

# paleta categorica validada (slots 1, 2, 3 -- os tres passam all-pairs)
SURFACE, TEXT_1, TEXT_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2de"

REGIMES = {
    "controle":  ("f6_ctrlpar_seed*", "Controle (100% dos dados)"),
    "ruido15":   ("f6_ruido15_seed*", "Ruído 15%"),
    # f6_ruido30 = semente 42 (rodada antes de o nome levar a semente; noise_seed=None)
    "ruido30":   ("f6_ruido30*",      "Ruído 30%"),
    "ruido50":   ("f6_ruido50_seed*", "Ruído 50%"),
    "frac10":    ("f6_frac10_seed*",  "Subamostragem 10%"),
    "frac02":    ("f6_frac02_seed*",  "Subamostragem 2%"),
    "frac01":    ("f6_frac01_seed*",  "Subamostragem 1%"),
    # passos igualados ao controle (~22.140): scripts/subamostragem_passos_igualados.sh
    "frac10eq":  ("f6_frac10eq_seed*", "Subamostragem 10% (passos igualados)"),
    "frac02eq":  ("f6_frac02eq_seed*", "Subamostragem 2% (passos igualados)"),
    "frac01eq":  ("f6_frac01eq_seed*", "Subamostragem 1% (passos igualados)"),
    # ruido ROSA fixo por imagem nas imagens de treino (src/data/image_noise.py)
    "rosa005":   ("f9_rosa_s005_seed*", "Ruído rosa σ = 0,05"),
    "rosa010":   ("f9_rosa_s010_seed*", "Ruído rosa σ = 0,1"),
    "rosa020":   ("f9_rosa_s020_seed*", "Ruído rosa σ = 0,2"),
    "branco005": ("f9_branco_s005_seed*", "Ruído branco σ = 0,05"),
    "branco010": ("f9_branco_s010_seed*", "Ruído branco σ = 0,1"),
    "branco020": ("f9_branco_s020_seed*", "Ruído branco σ = 0,2"),
    # DenseNet-121: camada densa classifier.weight (6 x 1.024); comparada com o controle DELA
    "dn_controle": ("dn121_ctrlpar_seed*", "DenseNet-121 · controle"),
    "dn_ruido30":  ("dn121_ruido30_seed*", "DenseNet-121 · ruído 30%"),
}
# cada regime e comparado com o controle da propria arquitetura
CONTROLE_DE = {r: ("dn_controle" if r.startswith("dn_") else "controle") for r in REGIMES}
# figura principal: um representante de cada familia, cores fixas por entidade
PRINCIPAIS = [("controle", "#2a78d6"), ("ruido30", "#eb6834"), ("frac02eq", "#1baf7a")]

ESTATISTICAS = [("lmc", "LMC"), ("sampen", "SampEn 1D"), ("sampen2d", "SampEn2D")]


def amplitude_relativa(v) -> float:
    """(max - min) / |media| -- mesma definicao de scripts/analise_f6.py."""
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    return float((v.max() - v.min()) / abs(v.mean())) if len(v) > 1 and v.mean() else float("nan")


def carregar(root: str) -> dict[str, list[pd.DataFrame]]:
    dados: dict[str, list[pd.DataFrame]] = {}
    for chave, (padrao, _) in REGIMES.items():
        runs = []
        for d in sorted(glob.glob(os.path.join(root, padrao))):
            if not os.path.isdir(d):
                continue
            df = pd.read_csv(os.path.join(d, "complexity_live.csv")).sort_values("epoch")
            cfg = json.load(open(os.path.join(d, "config.json")))
            passos_ep = math.ceil(cfg["particao"]["n_treino"] / cfg["batch_size"])
            hist = pd.read_csv(os.path.join(d, "history.csv"))
            df = df.drop(columns=[c for c in ("val_accuracy", "val_loss") if c in df.columns])
            df = df.merge(hist[["epoch", "val_loss", "val_accuracy", "train_eval_accuracy"]],
                          on="epoch", how="left")
            df["passos"] = df["epoch"] * passos_ep
            df["run"] = os.path.basename(d)
            df["n_treino"] = cfg["particao"]["n_treino"]
            runs.append(df.reset_index(drop=True))
        dados[chave] = runs
    return dados


def resumo(dados: dict) -> pd.DataFrame:
    linhas = []
    for chave, runs in dados.items():
        for df in runs:
            ln = {"regime": chave, "run": df["run"].iloc[0], "n_treino": int(df["n_treino"].iloc[0]),
                  "n_epocas": len(df), "passos_totais": int(df["passos"].iloc[-1]),
                  "acc_treino_final": df["train_eval_accuracy"].iloc[-1],
                  "acc_val_final": df["val_accuracy"].iloc[-1],
                  "acc_val_max": df["val_accuracy"].max()}
            for k, _ in ESTATISTICAS:
                v = df[k].to_numpy()
                ln[f"{k}_inicial"] = v[0]
                ln[f"{k}_final"] = v[-1]
                ln[f"{k}_amplitude"] = amplitude_relativa(v)
                ln[f"{k}_epoca_pico"] = int(df["epoch"].iloc[np.nanargmax(v)])
                ln[f"{k}_epoca_vale"] = int(df["epoch"].iloc[np.nanargmin(v)])
            linhas.append(ln)
    return pd.DataFrame(linhas)


def comparacoes(tab: pd.DataFrame) -> pd.DataFrame:
    """Cada regime contra o controle (Mann-Whitney bilateral). Descritivo, nao detector."""
    linhas = []
    for reg in [r for r in REGIMES if r not in CONTROLE_DE.values()]:
        ctrl = tab[tab.regime == CONTROLE_DE[reg]]
        g = tab[tab.regime == reg]
        for k, rot in ESTATISTICAS:
            for medida in ("final", "amplitude"):
                col = f"{k}_{medida}"
                a, b = g[col].to_numpy(), ctrl[col].to_numpy()
                p = mannwhitneyu(a, b, alternative="two-sided").pvalue
                linhas.append({"regime": reg, "controle": CONTROLE_DE[reg],
                               "estatistica": rot, "medida": medida,
                               "n": len(a), "media_regime": a.mean(), "dp_regime": a.std(ddof=1),
                               "media_controle": b.mean(), "dp_controle": b.std(ddof=1),
                               "razao": a.mean() / b.mean(), "p_mann_whitney": p})
    return pd.DataFrame(linhas)


def _eixo(ax, xlabel, ylabel, titulo):
    ax.set_facecolor(SURFACE)
    ax.set_xlabel(xlabel, color=TEXT_2, fontsize=9)
    ax.set_ylabel(ylabel, color=TEXT_2, fontsize=9)
    ax.set_title(titulo, color=TEXT_1, fontsize=10.5, loc="left", pad=8)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=TEXT_2, labelsize=8.5)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(GRID)


def _media_banda(runs, k, eixo):
    """Media e desvio entre sementes. Eixo de passos: interpola numa grade log comum."""
    if eixo == "epoch":
        n = min(len(r) for r in runs)
        x = runs[0]["epoch"].to_numpy()[:n]
        M = np.vstack([r[k].to_numpy()[:n] for r in runs])
    else:
        lo = max(r["passos"].iloc[0] for r in runs)
        hi = min(r["passos"].iloc[-1] for r in runs)
        x = np.geomspace(lo, hi, 120)
        M = np.vstack([np.interp(np.log(x), np.log(r["passos"]), r[k]) for r in runs])
    return x, M.mean(0), M.std(0)


def figura_trajetorias(dados, k, rotulo, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURFACE)
    for ax, eixo, xl in ((axes[0], "epoch", "Época"), (axes[1], "passos", "Passos de otimização (log)")):
        for chave, cor in PRINCIPAIS:
            x, m, s = _media_banda(dados[chave], k, eixo)
            ax.fill_between(x, m - s, m + s, color=cor, alpha=0.15, linewidth=0)
            ax.plot(x, m, color=cor, linewidth=2, label=REGIMES[chave][1])
        if eixo == "passos":
            ax.set_xscale("log")
        _eixo(ax, xl, rotulo, f"{'AB'[eixo == 'passos']}. {rotulo} por {xl.split(' (')[0].lower()}")
        ax.legend(frameon=False, fontsize=8.5, labelcolor=TEXT_2)
    fig.suptitle(f"{rotulo} da camada densa (fc.weight, 3.072 pesos) — média ± 1 DP entre sementes",
                 color=TEXT_1, fontsize=12, x=0.006, ha="left", weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def _painel_trajetorias(ax, dados, series, k, rotulo, eixo):
    for chave, cor, nome in series:
        x, m, s = _media_banda(dados[chave], k, eixo)
        ax.fill_between(x, m - s, m + s, color=cor, alpha=0.15, linewidth=0)
        ax.plot(x, m, color=cor, linewidth=2, label=nome)
    if eixo == "passos":
        ax.set_xscale("log")
    xl = "Passos de otimização (log)" if eixo == "passos" else "Época"
    _eixo(ax, xl, rotulo, rotulo)


def figura_niveis_ruido(dados, out):
    """Controle + ruido 15/30/50 %: o nivel de ruido e magnitude -> uma so matiz, clara a escura."""
    series = [("controle", "#7c7b76", "Controle (0%)"), ("ruido15", "#f4a37f", "Ruído 15%"),
              ("ruido30", "#eb6834", "Ruído 30%"), ("ruido50", "#a83a12", "Ruído 50%")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), facecolor=SURFACE)
    for ax, (k, rot) in zip(axes, ESTATISTICAS):
        _painel_trajetorias(ax, dados, series, k, rot, "epoch")
    axes[0].legend(frameon=False, fontsize=8.5, labelcolor=TEXT_2)
    fig.suptitle("Níveis de ruído nos rótulos — ResNet-18, camada densa, média ± 1 DP entre sementes",
                 color=TEXT_1, fontsize=12, x=0.006, ha="left", weight="bold")
    fig.text(0.006, 0.91, "Controle n = 8 · ruído 15% n = 3 · ruído 30% n = 8 · ruído 50% n = 3. "
             "A queda da SampEn não cresce com o nível de ruído.", color=TEXT_2, fontsize=9, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def figura_ruido_rosa(dados, out):
    """Controle + ruido rosa nas imagens, sigma 0,05/0,1/0,2: magnitude -> uma matiz, clara a escura."""
    series = [("controle", "#7c7b76", "Controle (sem ruído)"), ("rosa005", "#f2b6cc", "Rosa σ = 0,05"),
              ("rosa010", "#e87ba4", "Rosa σ = 0,1"), ("rosa020", "#a8335f", "Rosa σ = 0,2")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), facecolor=SURFACE)
    for ax, (k, rot) in zip(axes, ESTATISTICAS):
        _painel_trajetorias(ax, dados, series, k, rot, "epoch")
    axes[0].legend(frameon=False, fontsize=8.5, labelcolor=TEXT_2)
    fig.suptitle("Ruído rosa nas imagens de treino — ResNet-18, camada densa, média ± 1 DP entre sementes",
                 color=TEXT_1, fontsize=12, x=0.006, ha="left", weight="bold")
    fig.text(0.006, 0.91, "Controle n = 8 · cada intensidade n = 5. Ruído fixo por imagem, só no treino; "
             "validação e teste limpos.", color=TEXT_2, fontsize=9, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def figura_ruido_branco(dados, out):
    """Mesmo desenho de figura_ruido_rosa, para o ruido branco."""
    series = [("controle", "#2a78d6", "Controle (sem ruído)"), ("branco005", "#c3c2b7", "Branco σ = 0,05"),
              ("branco010", "#8a8980", "Branco σ = 0,1"), ("branco020", "#3d3c38", "Branco σ = 0,2")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), facecolor=SURFACE)
    for ax, (k, rot) in zip(axes, ESTATISTICAS):
        _painel_trajetorias(ax, dados, series, k, rot, "epoch")
    axes[0].legend(frameon=False, fontsize=8.5, labelcolor=TEXT_2)
    fig.suptitle("Ruído branco nas imagens de treino — ResNet-18, camada densa, média ± 1 DP entre sementes",
                 color=TEXT_1, fontsize=12, x=0.006, ha="left", weight="bold")
    fig.text(0.006, 0.91, "Controle n = 8 · cada intensidade n = 5. Ruído fixo por imagem, só no treino; "
             "validação e teste limpos.", color=TEXT_2, fontsize=9, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def figura_densenet(dados, out):
    """DenseNet-121: controle x ruido 30 %, as tres estatisticas por epoca."""
    series = [("dn_controle", "#2a78d6", "Controle"), ("dn_ruido30", "#eb6834", "Ruído 30%")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), facecolor=SURFACE)
    for ax, (k, rot) in zip(axes, ESTATISTICAS):
        _painel_trajetorias(ax, dados, series, k, rot, "epoch")
    axes[0].legend(frameon=False, fontsize=8.5, labelcolor=TEXT_2)
    n_c, n_r = len(dados["dn_controle"]), len(dados["dn_ruido30"])
    fig.suptitle("DenseNet-121 — camada densa (classifier.weight, 6.144 pesos), média ± 1 DP entre sementes",
                 color=TEXT_1, fontsize=12, x=0.006, ha="left", weight="bold")
    fig.text(0.006, 0.91, f"Controle n = {n_c} · ruído 30% n = {n_r}. 60 épocas, 369 passos por época.",
             color=TEXT_2, fontsize=9, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description="Comportamento de LMC/SampEn na camada densa")
    p.add_argument("--root", default="monai_weights")
    p.add_argument("--output_dir", default="monai_weights/f8_comportamento")
    a = p.parse_args()
    os.makedirs(a.output_dir, exist_ok=True)

    dados = carregar(a.root)
    tab = resumo(dados)
    comp = comparacoes(tab)
    tab.to_csv(os.path.join(a.output_dir, "comportamento_por_run.csv"), index=False)
    comp.to_csv(os.path.join(a.output_dir, "comparacoes_vs_controle.csv"), index=False)

    # trajetorias medias por regime, para a apresentacao
    trajs = []
    for chave, runs in dados.items():
        for k, _ in ESTATISTICAS:
            for eixo in ("epoch", "passos"):
                x, m, s = _media_banda(runs, k, eixo)
                trajs.append(pd.DataFrame({"regime": chave, "estatistica": k, "eixo": eixo,
                                           "x": x, "media": m, "dp": s}))
    pd.concat(trajs).to_csv(os.path.join(a.output_dir, "trajetorias_medias.csv"), index=False)

    for k, rot in ESTATISTICAS:
        figura_trajetorias(dados, k, rot, os.path.join(a.output_dir, f"trajetoria_{k}.png"))
    figura_niveis_ruido(dados, os.path.join(a.output_dir, "niveis_de_ruido.png"))
    figura_densenet(dados, os.path.join(a.output_dir, "densenet121_controle_vs_ruido.png"))
    figura_ruido_rosa(dados, os.path.join(a.output_dir, "ruido_rosa.png"))
    figura_ruido_branco(dados, os.path.join(a.output_dir, "ruido_branco.png"))

    pd.set_option("display.width", 200)
    print(tab.groupby("regime")[["n_treino", "passos_totais", "acc_treino_final", "acc_val_final",
                                 "lmc_inicial", "lmc_final", "lmc_amplitude",
                                 "sampen_final", "sampen_amplitude",
                                 "sampen2d_final", "sampen2d_amplitude"]].mean().round(4).to_string())
    print()
    print(comp.round(4).to_string(index=False))
    print(f"\nSalvo em {a.output_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())

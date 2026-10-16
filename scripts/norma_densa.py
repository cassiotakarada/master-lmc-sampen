"""
Norma dos pesos da camada densa como medida de comparacao (linha de base simples).

Pergunta que uma banca faz: a LMC ou a SampEn trazem algo que uma medida trivial -- o
"tamanho" dos pesos -- ja nao mostra? Este script calcula, epoca a epoca, a norma de
Frobenius ||W||_F da camada densa (fc.weight / classifier.weight) e:
  1. compara cada regime com o controle da sua arquitetura (valor final e amplitude,
     Mann-Whitney), na mesma forma de scripts/comportamento_densa.py;
  2. mede, dentro de cada treino, a correlacao de Spearman entre a norma e a LMC, a
     SampEn 1D e a SampEn2D ao longo das epocas.
Usa os checkpoints epoch_*.pth (os runs de passos igualados nao os tem e ficam de fora).
"""
from __future__ import annotations

import argparse
import glob
import os
import pathlib
import re
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor

warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

SURFACE, TEXT_1, TEXT_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2de"

REGIMES = [
    ("controle", "f6_ctrlpar_seed*"), ("ruido15", "f6_ruido15_seed*"), ("ruido30", "f6_ruido30*"),
    ("ruido50", "f6_ruido50_seed*"),
    ("frac10", "f6_frac10_seed*"), ("frac02", "f6_frac02_seed*"), ("frac01", "f6_frac01_seed*"),
    ("rosa005", "f9_rosa_s005_seed*"), ("rosa010", "f9_rosa_s010_seed*"), ("rosa020", "f9_rosa_s020_seed*"),
    ("branco005", "f9_branco_s005_seed*"), ("branco010", "f9_branco_s010_seed*"),
    ("branco020", "f9_branco_s020_seed*"),
    ("dn_controle", "dn121_ctrlpar_seed*"), ("dn_ruido30", "dn121_ruido30_seed*"),
]
CONTROLE_DE = {r: ("dn_controle" if r.startswith("dn_") else "controle") for r, _ in REGIMES}
FIGURA = [("controle", "#2a78d6", "Controle"), ("ruido30", "#eb6834", "Ruído 30% nos rótulos"),
          ("rosa020", "#e87ba4", "Ruído rosa σ = 0,2"), ("branco020", "#7c7b76", "Ruído branco σ = 0,2")]


def norma_run(args) -> pd.DataFrame:
    import torch
    regime, run_dir = args
    linhas = []
    for ck in glob.glob(os.path.join(run_dir, "epoch_*.pth")):
        sd = torch.load(ck, map_location="cpu", weights_only=True)
        if isinstance(sd, dict) and "model_state_dict" in sd:
            sd = sd["model_state_dict"]
        W = [v for k, v in sd.items() if torch.is_tensor(v) and v.ndim == 2 and k.endswith(".weight")][0]
        linhas.append({"epoch": int(re.search(r"epoch_(\d+)", ck).group(1)),
                       "norma": float(W.float().norm())})
    df = pd.DataFrame(linhas).sort_values("epoch")
    live = pd.read_csv(os.path.join(run_dir, "complexity_live.csv"))[["epoch", "lmc", "sampen", "sampen2d"]]
    df = df.merge(live, on="epoch", how="left")
    df.insert(0, "run", os.path.basename(run_dir))
    df.insert(0, "regime", regime)
    return df


def amplitude_relativa(v) -> float:
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    return float((v.max() - v.min()) / abs(v.mean())) if len(v) > 1 and v.mean() else float("nan")


def main() -> int:
    p = argparse.ArgumentParser(description="Norma da camada densa como linha de base")
    p.add_argument("--root", default="monai_weights")
    p.add_argument("--output_dir", default="monai_weights/f8_norma")
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    os.makedirs(a.output_dir, exist_ok=True)

    tarefas = []
    for regime, padrao in REGIMES:
        for d in sorted(x for x in glob.glob(os.path.join(a.root, padrao)) if os.path.isdir(x)):
            if os.path.isfile(os.path.join(d, "test_metrics.json")) and glob.glob(os.path.join(d, "epoch_*.pth")):
                tarefas.append((regime, d))
    with ProcessPoolExecutor(a.workers) as ex:
        df = pd.concat(ex.map(norma_run, tarefas), ignore_index=True)
    df.to_csv(os.path.join(a.output_dir, "norma_por_epoca.csv"), index=False)

    # por treino: final, amplitude e correlacao com as medidas de complexidade
    por_run = []
    for (regime, run), g in df.groupby(["regime", "run"]):
        ln = {"regime": regime, "run": run, "norma_final": g.norma.iloc[-1],
              "norma_amplitude": amplitude_relativa(g.norma)}
        for k in ("lmc", "sampen", "sampen2d"):
            ln[f"spearman_norma_{k}"] = spearmanr(g.norma, g[k]).correlation
        por_run.append(ln)
    por_run = pd.DataFrame(por_run)
    por_run.to_csv(os.path.join(a.output_dir, "norma_por_run.csv"), index=False)

    comps = []
    for regime, _ in REGIMES:
        ctrl = CONTROLE_DE[regime]
        if regime == ctrl or regime not in set(por_run.regime):
            continue
        g, c = por_run[por_run.regime == regime], por_run[por_run.regime == ctrl]
        for medida in ("norma_final", "norma_amplitude"):
            comps.append({"regime": regime, "controle": ctrl, "medida": medida, "n": len(g),
                          "media_regime": g[medida].mean(), "media_controle": c[medida].mean(),
                          "razao": g[medida].mean() / c[medida].mean(),
                          "p_mann_whitney": mannwhitneyu(g[medida], c[medida]).pvalue})
    comps = pd.DataFrame(comps)
    comps.to_csv(os.path.join(a.output_dir, "norma_vs_controle.csv"), index=False)

    corr = por_run.groupby("regime")[[c for c in por_run if c.startswith("spearman")]].mean()
    corr.to_csv(os.path.join(a.output_dir, "correlacao_norma_complexidade.csv"))

    # figura: trajetoria da norma + norma x LMC
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), facecolor=SURFACE)
    for chave, cor, nome in FIGURA:
        sub = df[df.regime == chave]
        if sub.empty:
            continue
        m = sub.groupby("epoch").norma.mean()
        s = sub.groupby("epoch").norma.std()
        axes[0].fill_between(m.index, m - s, m + s, color=cor, alpha=0.15, linewidth=0)
        axes[0].plot(m.index, m, color=cor, linewidth=2, label=nome)
        axes[1].scatter(sub.norma, sub.lmc, s=6, color=cor, alpha=0.5, label=nome)
    titulos = [("Época", "‖W‖ (norma de Frobenius)", "A. Norma da camada densa por época"),
               ("‖W‖", "LMC", "B. LMC contra a norma (cada ponto = uma época de um treino)")]
    for ax, (xl, yl, t) in zip(axes, titulos):
        ax.set_facecolor(SURFACE)
        ax.set_xlabel(xl, color=TEXT_2, fontsize=9)
        ax.set_ylabel(yl, color=TEXT_2, fontsize=9)
        ax.set_title(t, color=TEXT_1, fontsize=10.5, loc="left")
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.tick_params(colors=TEXT_2, labelsize=8.5)
        for lado in ("top", "right"):
            ax.spines[lado].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8.5, labelcolor=TEXT_2)
    fig.suptitle("Linha de base: norma dos pesos da camada densa — ResNet-18",
                 color=TEXT_1, fontsize=12, x=0.006, ha="left", weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(a.output_dir, "norma_densa.png"), dpi=150, facecolor=SURFACE)

    pd.set_option("display.width", 200)
    print(comps.round(4).to_string(index=False))
    print()
    print(corr.round(3).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())

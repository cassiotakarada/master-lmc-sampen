"""
Sensibilidade aos parametros das medidas, nos runs ATUAIS (ResNet-18, camada densa).

A pergunta e descritiva: o COMPORTAMENTO relatado (LMC sobe com o treino; SampEn cai sob
ruido de rotulo e fica estavel no controle) se mantem quando mudamos o parametro livre de
cada medida? O valor absoluto muda -- isso e esperado e nao e o ponto.

  LMC        n_bins  em {25, 50, 100, 200}        (100 = valor usado no trabalho)
  SampEn 1D  r_scale em {0,10; 0,15; 0,20; 0,25}  (m = 2, ordem por neuronio)
  SampEn2D   r_scale em {0,10; 0,15; 0,20; 0,25}  (m = 1, matriz 6 x 512 sem subamostragem)

Runs: controle pareado (f6_ctrlpar_seed*, n = 8) x ruido 30 % (f6_ruido30*, n = 8), 60 epocas,
pesos lidos dos checkpoints epoch_*.pth. Substitui monai_weights/sensibilidade/, feita em
runs legados do MONAI.
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
from scipy.stats import mannwhitneyu

from src.complexity import flatten_dense, lmc_complexity, sample_entropy_1d, sample_entropy_2d

SURFACE, TEXT_1, TEXT_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2de"
COR = {"controle": "#2a78d6", "ruido30": "#eb6834"}
GRUPOS = {"controle": "f6_ctrlpar_seed*", "ruido30": "f6_ruido30*"}
BINS = [25, 50, 100, 200]
RS = [0.10, 0.15, 0.20, 0.25]
MEDIDAS = ([("lmc", b, f"LMC · {b} bins") for b in BINS]
           + [("sampen", r, f"SampEn 1D · r = {r:.2f}·DP") for r in RS]
           + [("sampen2d", r, f"SampEn2D · r = {r:.2f}·DP") for r in RS])


def peso_denso(caminho: str) -> np.ndarray:
    import torch
    sd = torch.load(caminho, map_location="cpu", weights_only=True)
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    densas = [v for k, v in sd.items() if torch.is_tensor(v) and v.ndim == 2 and k.endswith(".weight")]
    return densas[0].float().numpy()


def medir_run(args) -> pd.DataFrame:
    grupo, run_dir = args
    linhas = []
    for ck in sorted(glob.glob(os.path.join(run_dir, "epoch_*.pth"))):
        ep = int(re.search(r"epoch_(\d+)", ck).group(1))
        W = peso_denso(ck)
        v = flatten_dense(W, order="n_major")
        for medida, par, _ in MEDIDAS:
            if medida == "lmc":
                val = lmc_complexity(v, n_bins=par)["complexity"]
            elif medida == "sampen":
                val = sample_entropy_1d(v, m=2, r_scale=par)
            else:
                val = sample_entropy_2d(W, m=1, r_scale=par, max_dim=int(max(W.shape)))
            linhas.append({"grupo": grupo, "run": os.path.basename(run_dir), "epoch": ep,
                           "medida": medida, "parametro": par,
                           "valor": float(val) if np.isfinite(val) else np.nan})
    return pd.DataFrame(linhas)


def amplitude_relativa(v) -> float:
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    return float((v.max() - v.min()) / abs(v.mean())) if len(v) > 1 and v.mean() else float("nan")


def resumo(df: pd.DataFrame) -> pd.DataFrame:
    por_run = (df.sort_values("epoch").groupby(["medida", "parametro", "grupo", "run"])["valor"]
               .agg(final="last", amplitude=amplitude_relativa).reset_index())
    linhas = []
    for (med, par), g in por_run.groupby(["medida", "parametro"]):
        c, r = g[g.grupo == "controle"], g[g.grupo == "ruido30"]
        for qual in ("final", "amplitude"):
            linhas.append({"medida": med, "parametro": par, "comparacao": qual,
                           "controle_media": c[qual].mean(), "ruido30_media": r[qual].mean(),
                           "razao_ruido_controle": r[qual].mean() / c[qual].mean(),
                           "p_mann_whitney": mannwhitneyu(r[qual], c[qual]).pvalue})
    return pd.DataFrame(linhas)


def figura(df: pd.DataFrame, res: pd.DataFrame, out: str) -> None:
    fig, axes = plt.subplots(3, 4, figsize=(17, 11), facecolor=SURFACE)
    for ax, (med, par, titulo) in zip(axes.ravel(), MEDIDAS):
        sub = df[(df.medida == med) & (df.parametro == par)]
        for grupo, nome in (("controle", "Controle"), ("ruido30", "Ruído 30%")):
            g = sub[sub.grupo == grupo].groupby("epoch")["valor"]
            m, s = g.mean(), g.std()
            ax.fill_between(m.index, m - s, m + s, color=COR[grupo], alpha=0.15, linewidth=0)
            ax.plot(m.index, m, color=COR[grupo], linewidth=1.8, label=nome)
        r = res[(res.medida == med) & (res.parametro == par) & (res.comparacao == "amplitude")].iloc[0]
        padrao = " (usado)" if (med == "lmc" and par == 100) or (med != "lmc" and abs(par - 0.2) < 1e-9) else ""
        ax.set_facecolor(SURFACE)
        ax.set_title(f"{titulo}{padrao}", color=TEXT_1, fontsize=10.5, loc="left")
        ax.text(0.02, 0.04, f"amplitude ruído/controle = {r.razao_ruido_controle:.2f}×  ·  p = {r.p_mann_whitney:.4f}",
                transform=ax.transAxes, color=TEXT_2, fontsize=8.5)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.tick_params(colors=TEXT_2, labelsize=8)
        ax.set_xlabel("Época", color=TEXT_2, fontsize=8.5)
        for lado in ("top", "right"):
            ax.spines[lado].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=9, labelcolor=TEXT_2)
    fig.suptitle("Sensibilidade aos parâmetros — ResNet-18, camada densa, controle (n = 8) × ruído 30% (n = 8)",
                 color=TEXT_1, fontsize=13, x=0.006, ha="left", weight="bold")
    fig.text(0.006, 0.935, "Cada linha muda um parâmetro. O valor absoluto muda; a pergunta é se o desenho "
             "das curvas e a separação entre os grupos se mantêm.", color=TEXT_2, fontsize=9.5, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description="Sensibilidade de LMC/SampEn na camada densa")
    p.add_argument("--root", default="monai_weights")
    p.add_argument("--output_dir", default="monai_weights/f8_sensibilidade")
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    os.makedirs(a.output_dir, exist_ok=True)

    tarefas = [(g, d) for g, padrao in GRUPOS.items()
               for d in sorted(glob.glob(os.path.join(a.root, padrao))) if os.path.isdir(d)]
    with ProcessPoolExecutor(a.workers) as ex:
        df = pd.concat(ex.map(medir_run, tarefas), ignore_index=True)
    res = resumo(df)
    df.to_csv(os.path.join(a.output_dir, "sensibilidade_por_epoca.csv"), index=False)
    res.to_csv(os.path.join(a.output_dir, "sensibilidade_resumo.csv"), index=False)
    figura(df, res, os.path.join(a.output_dir, "sensibilidade_parametros.png"))
    pd.set_option("display.width", 200)
    print(res.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

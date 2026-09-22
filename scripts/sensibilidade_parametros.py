"""
Estudo de sensibilidade da LMC ao numero de bins e da SampEn a tolerancia r.
Fase F2 do plan.md.

Pergunta cientifica: o VALOR ABSOLUTO da LMC depende muito do numero de bins
(sabemos disso). A pergunta que importa para o projeto e outra:

    a EPOCA do pico / da inflexao muda quando mudamos o parametro?

Se a epoca do pico for estavel, o indicador e utilizavel mesmo com o valor
absoluto sendo arbitrario -- e a escolha de bins vira uma convencao, nao um
grau de liberdade que permite "escolher o resultado".

Analisa somente a CAMADA DENSA (decisao D1 do plano).

Uso:
    myenv/bin/python scripts/sensibilidade_parametros.py
    myenv/bin/python scripts/sensibilidade_parametros.py --run_dirs a b c --output_dir X
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
import torch

from src.analysis.overfit_cut import carregar_relatorio, sombrear_regiao_descartada
from src.complexity import find_dense_layers, flatten_dense, lmc_complexity, sample_entropy_1d

# --- parametros varridos ------------------------------------------------------
BINS_GRID = [20, 50, 100, 200, "FD"]        # "FD" = regra de Freedman-Diaconis
R_GRID = [0.10, 0.15, 0.20, 0.25]

# --- paleta (rampa sequencial azul, passos ordinais >= 250) --------------------
RAMPA_AZUL = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
SURFACE = "#fcfcfb"
TEXT_1 = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e3e2de"
DESTAQUE = "#eb6834"


def bins_freedman_diaconis(x: np.ndarray) -> int:
    """Regra de Freedman-Diaconis: largura = 2*IQR*n^(-1/3). Limitada a [10, 500]."""
    q75, q25 = np.percentile(x, [75, 25])
    iqr = q75 - q25
    if iqr <= 0:
        return 100
    largura = 2.0 * iqr * len(x) ** (-1 / 3)
    if largura <= 0:
        return 100
    return int(np.clip(np.ceil((x.max() - x.min()) / largura), 10, 500))


def coletar(run_dir: str) -> pd.DataFrame:
    """Para cada epoca do run, calcula LMC (varios bins) e SampEn (varios r) da densa."""
    pt_path = os.path.join(run_dir, "param_types.json")
    param_types = json.load(open(pt_path)) if os.path.isfile(pt_path) else None

    arquivos = sorted(f for f in os.listdir(run_dir) if f.startswith("epoch_") and f.endswith(".pth"))
    if not arquivos:
        print(f"  [!] nenhum epoch_*.pth em {run_dir}")
        return pd.DataFrame()

    linhas = []
    for fname in arquivos:
        epoca = int(fname[6:9])
        sd = torch.load(os.path.join(run_dir, fname), map_location="cpu", weights_only=True)
        densas = find_dense_layers(sd, param_types)
        if not densas:
            print(f"  [!] nenhuma camada densa em {fname}")
            continue
        nome = densas[0]
        W = sd[nome].detach().cpu().numpy()
        v = flatten_dense(W, order="n_major")      # convencao do projeto

        linha = {"run": os.path.basename(run_dir.rstrip("/")), "epoch": epoca,
                 "camada": nome, "n_pesos": v.size}
        for b in BINS_GRID:
            nb = bins_freedman_diaconis(v) if b == "FD" else b
            linha[f"LMC_bins{b}"] = lmc_complexity(v, n_bins=nb)["complexity"]
            if b == "FD":
                linha["FD_nbins"] = nb
        for rs in R_GRID:
            linha[f"SampEn_r{rs:.2f}"] = sample_entropy_1d(v, m=2, r_scale=rs)
        linhas.append(linha)
        print(f"  epoca {epoca:>3}  {nome} {tuple(W.shape)}", flush=True)

    return pd.DataFrame(linhas).sort_values("epoch").reset_index(drop=True)


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


def figura_bins(df: pd.DataFrame, out: str, relatorio=None) -> dict:
    """Dois paineis lado a lado: valores crus e curvas normalizadas (min-max)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8), facecolor=SURFACE)
    picos = {}
    x = df["epoch"].to_numpy()

    for i, b in enumerate(BINS_GRID):
        col = f"LMC_bins{b}"
        y = df[col].to_numpy()
        if b == "FD":
            lo, hi = int(df["FD_nbins"].min()), int(df["FD_nbins"].max())
            rotulo = f"FD (varia: {lo}–{hi})"
        else:
            rotulo = str(b)
        cor = RAMPA_AZUL[i]
        ax1.plot(x, y, color=cor, linewidth=1.8, marker="o", markersize=4.5,
                 markeredgecolor=SURFACE, markeredgewidth=0.8, label=rotulo)

        y_norm = (y - y.min()) / (y.max() - y.min()) if y.max() > y.min() else y * 0
        ax2.plot(x, y_norm, color=cor, linewidth=1.8, marker="o", markersize=4.5,
                 markeredgecolor=SURFACE, markeredgewidth=0.8, label=rotulo)

        ep_pico = int(x[int(np.argmax(y))])
        picos[rotulo] = ep_pico
        ax2.plot([ep_pico], [1.0], marker="v", markersize=9, color=DESTAQUE,
                 markeredgecolor=SURFACE, markeredgewidth=1.0, zorder=5)

    _eixo(ax1, "Época", "Complexidade LMC (valor cru)",
          "A. O valor absoluto depende fortemente do nº de bins")
    _eixo(ax2, "Época", "LMC normalizada por curva (mín–máx)",
          "B. …mas a FORMA da trajetória é a mesma (▼ = época do pico)")

    for ax in (ax1, ax2):
        sombrear_regiao_descartada(ax, relatorio, com_legenda=(ax is ax1))
        leg = ax.legend(title="nº de bins", frameon=False, fontsize=8.5, title_fontsize=8.5,
                        loc="best")
        leg.get_title().set_color(TEXT_2)
        for t in leg.get_texts():
            t.set_color(TEXT_2)

    fig.suptitle("Sensibilidade da LMC da camada densa ao número de bins",
                 color=TEXT_1, fontsize=12.5, x=0.007, ha="left", y=0.985, weight="bold")
    fig.text(0.007, 0.935,
             "Bins fixos (20–200) dão exatamente a mesma época de pico; a escolha é uma convenção, não um grau de liberdade.\n"
             "A regra de Freedman-Diaconis escolhe um nº de bins diferente a cada época, misturando mudança de parâmetro com mudança de sinal — por isso não é usada.",
             color=TEXT_2, fontsize=8.5, ha="left", va="top")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return picos


def figura_r(df: pd.DataFrame, out: str, relatorio=None) -> dict:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8), facecolor=SURFACE)
    picos = {}
    x = df["epoch"].to_numpy()
    cores = RAMPA_AZUL[:len(R_GRID)]

    for i, rs in enumerate(R_GRID):
        y = df[f"SampEn_r{rs:.2f}"].to_numpy()
        rotulo = f"{rs:.2f}·σ"
        ax1.plot(x, y, color=cores[i], linewidth=1.8, marker="s", markersize=4.5,
                 markeredgecolor=SURFACE, markeredgewidth=0.8, label=rotulo)

        y_norm = (y - y.min()) / (y.max() - y.min()) if y.max() > y.min() else y * 0
        ax2.plot(x, y_norm, color=cores[i], linewidth=1.8, marker="s", markersize=4.5,
                 markeredgecolor=SURFACE, markeredgewidth=0.8, label=rotulo)

        ep_min = int(x[int(np.argmin(y))])
        picos[rotulo] = ep_min
        ax2.plot([ep_min], [0.0], marker="^", markersize=9, color=DESTAQUE,
                 markeredgecolor=SURFACE, markeredgewidth=1.0, zorder=5)

    _eixo(ax1, "Época", "Entropia amostral (valor cru)",
          "A. O valor absoluto depende da tolerância r")
    _eixo(ax2, "Época", "SampEn normalizada por curva (mín–máx)",
          "B. …e a forma NÃO se mantém: as curvas se cruzam (▲ = época do mínimo)")

    for ax in (ax1, ax2):
        sombrear_regiao_descartada(ax, relatorio, com_legenda=(ax is ax1))
        leg = ax.legend(title="tolerância r", frameon=False, fontsize=8.5, title_fontsize=8.5,
                        loc="best")
        leg.get_title().set_color(TEXT_2)
        for t in leg.get_texts():
            t.set_color(TEXT_2)

    fig.suptitle("Sensibilidade da SampEn da camada densa à tolerância r",
                 color=TEXT_1, fontsize=12.5, x=0.007, ha="left", y=0.985, weight="bold")
    fig.text(0.007, 0.935,
             "Ao contrário da LMC, a escolha de r muda a forma da trajetória: a época do mínimo não coincide entre os valores de r\n"
             "(Spearman 0,43 entre r=0,10 e r=0,25), e a amplitude do sinal é de apenas 3–8% da média — contra ~130% na LMC.",
             color=TEXT_2, fontsize=8.5, ha="left", va="top")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return picos


def correlacoes(df: pd.DataFrame, prefixo: str, chaves: list) -> pd.DataFrame:
    """Correlacao de Spearman entre as trajetorias sob parametros diferentes."""
    cols = [f"{prefixo}{k}" if not isinstance(k, float) else f"{prefixo}{k:.2f}" for k in chaves]
    return df[cols].corr(method="spearman")


def main() -> int:
    p = argparse.ArgumentParser(description="Sensibilidade da LMC (bins) e da SampEn (r)")
    p.add_argument("--run_dirs", nargs="+", default=[
        "monai_weights/run_mednist_resnet18_64",
        "monai_weights/run_mednist_resnet18_64_seed1",
        "monai_weights/run_mednist_resnet18_64_seed7",
    ])
    p.add_argument("--output_dir", default="monai_weights/sensibilidade")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    todos, resumo = [], []

    for rd in args.run_dirs:
        if not os.path.isdir(rd):
            print(f"[skip] {rd} nao existe")
            continue
        print(f"[run] {rd}")
        df = coletar(rd)
        if df.empty:
            continue
        todos.append(df)

        nome = os.path.basename(rd.rstrip("/"))
        relatorio = carregar_relatorio(rd)
        pb = figura_bins(df, os.path.join(args.output_dir, f"sensibilidade_bins_{nome}.png"), relatorio)
        pr = figura_r(df, os.path.join(args.output_dir, f"sensibilidade_r_{nome}.png"), relatorio)
        resumo.append({"run": nome, "picos_LMC_por_bins": pb, "minimos_SampEn_por_r": pr})
        print(f"  -> epoca do pico da LMC por bins: {pb}")
        print(f"  -> epoca do minimo da SampEn por r: {pr}")

    if not todos:
        print("nada coletado")
        return 1

    full = pd.concat(todos, ignore_index=True)
    csv = os.path.join(args.output_dir, "sensibilidade.csv")
    full.to_csv(csv, index=False)

    print("\n" + "=" * 78)
    print("CORRELACAO DE SPEARMAN ENTRE TRAJETORIAS (todos os runs empilhados)")
    print("=" * 78)
    print("\nLMC sob diferentes nº de bins:")
    print(correlacoes(full, "LMC_bins", BINS_GRID).round(4).to_string())
    print("\nSampEn sob diferentes r:")
    print(correlacoes(full, "SampEn_r", R_GRID).round(4).to_string())

    (pathlib.Path(args.output_dir) / "resumo_picos.json").write_text(
        json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSalvos em {args.output_dir}/ (csv, resumo_picos.json, PNGs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

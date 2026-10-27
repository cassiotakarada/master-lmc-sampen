"""
MedNIST x PathMNIST: os efeitos de cada regime sobre a LMC/SampEn da camada densa se repetem?

Le as tabelas que `scripts/comportamento_densa.py` grava para cada dataset
(comparacoes_vs_controle.csv e comportamento_por_run.csv) e produz:

  comparacao_regimes.csv   razao regime/controle e p de Mann-Whitney nos dois datasets,
                           com a classificacao do padrao (repete, so num dataset, diverge)
  densenet_vs_resnet.csv   controle DenseNet-121 x controle ResNet-18, em cada dataset
  batchnorm_vs_medidas.csv nos controles: a oscilacao da acuracia entre epocas (BatchNorm
                           em eval) acompanha a variacao da LMC/SampEn? Spearman por run
  comparacao_datasets.png  razao regime/controle nos dois datasets, por medida

Descritivo, como o resto do projeto: p < 0,05 sem correcao para multiplos testes, entao
um efeito isolado nao deve ser lido como achado.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

SURFACE, TEXT_1, TEXT_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2de"
COR = {"mednist": "#2a78d6", "pathmnist": "#eb6834"}
NOME = {"mednist": "MedNIST", "pathmnist": "PathMNIST"}
DIRS = {"mednist": "monai_weights/f8_comportamento", "pathmnist": "monai_weights/pm_comportamento"}
CONTROLES = {"mednist": ("f6_ctrlpar_seed*", "dn121_ctrlpar_seed*"),
             "pathmnist": ("pm_ctrl_seed*", "pm_dn121_ctrl_seed*")}
ALFA = 0.05

ROTULO_REGIME = {
    "frac10": "Subamostragem 10%", "frac02": "Subamostragem 2%", "frac01": "Subamostragem 1%",
    "frac10eq": "10%, passos igualados", "frac02eq": "2%, passos igualados",
    "frac01eq": "1%, passos igualados",
    "rosa005": "Rosa σ = 0,05", "rosa010": "Rosa σ = 0,1", "rosa020": "Rosa σ = 0,2",
    "branco005": "Branco σ = 0,05", "branco010": "Branco σ = 0,1", "branco020": "Branco σ = 0,2",
}


def padrao(r_m, p_m, r_p, p_p) -> str:
    sig_m, sig_p = p_m < ALFA, p_p < ALFA
    if sig_m and sig_p:
        return "repete" if np.sign(r_m - 1) == np.sign(r_p - 1) else "diverge"
    if sig_m:
        return "so MedNIST"
    if sig_p:
        return "so PathMNIST"
    return "nenhum"


def comparar_regimes() -> pd.DataFrame:
    tabs = {}
    for ds, d in DIRS.items():
        t = pd.read_csv(os.path.join(d, "comparacoes_vs_controle.csv"))
        tabs[ds] = t[["regime", "estatistica", "medida", "razao", "p_mann_whitney"]].rename(
            columns={"razao": f"razao_{ds}", "p_mann_whitney": f"p_{ds}"})
    m = tabs["mednist"].merge(tabs["pathmnist"], on=["regime", "estatistica", "medida"])
    m["padrao"] = [padrao(*v) for v in m[["razao_mednist", "p_mednist",
                                          "razao_pathmnist", "p_pathmnist"]].to_numpy()]
    return m


def densenet_vs_resnet() -> pd.DataFrame:
    linhas = []
    for ds, d in DIRS.items():
        t = pd.read_csv(os.path.join(d, "comportamento_por_run.csv"))
        rn, dn = t[t.regime == "controle"], t[t.regime == "dn_controle"]
        for k, rot in (("lmc", "LMC"), ("sampen", "SampEn 1D")):
            for medida in ("final", "amplitude"):
                a, b = dn[f"{k}_{medida}"].to_numpy(), rn[f"{k}_{medida}"].to_numpy()
                linhas.append({"dataset": NOME[ds], "estatistica": rot, "medida": medida,
                               "n_densenet": len(a), "n_resnet": len(b),
                               "media_densenet": a.mean(), "media_resnet": b.mean(),
                               "razao": a.mean() / b.mean(),
                               "p_mann_whitney": mannwhitneyu(a, b, alternative="two-sided").pvalue})
    return pd.DataFrame(linhas)


def batchnorm_vs_medidas(root: str) -> pd.DataFrame:
    """Spearman entre |delta acuracia de validacao| e |delta LMC| (e SampEn), epoca a epoca.

    Se a oscilacao da acuracia fosse acompanhada pelas medidas, as quedas bruscas de
    acuracia coincidiriam com saltos da LMC/SampEn. A LMC e a SampEn leem os PESOS da
    densa; o BatchNorm em eval mexe nas ATIVACOES, entao a expectativa e correlacao fraca.
    """
    linhas = []
    for ds, padroes in CONTROLES.items():
        for padrao_run, arq in zip(padroes, ("ResNet-18", "DenseNet-121")):
            for d in sorted(x for x in glob.glob(os.path.join(root, padrao_run)) if os.path.isdir(x)):
                live = pd.read_csv(os.path.join(d, "complexity_live.csv")).sort_values("epoch")
                hist = pd.read_csv(os.path.join(d, "history.csv")).sort_values("epoch")
                df = live[["epoch", "lmc", "sampen"]].merge(hist[["epoch", "val_accuracy"]], on="epoch")
                dacc = df["val_accuracy"].diff().abs().iloc[1:]
                ln = {"dataset": NOME[ds], "arquitetura": arq, "run": os.path.basename(d),
                      "dp_delta_acc_val": float(df["val_accuracy"].diff().std()),
                      "maior_queda_acc_val": float(-df["val_accuracy"].diff().min())}
                for k in ("lmc", "sampen"):
                    rho, p = spearmanr(dacc, df[k].diff().abs().iloc[1:])
                    ln[f"spearman_{k}"], ln[f"p_{k}"] = float(rho), float(p)
                linhas.append(ln)
    return pd.DataFrame(linhas)


def figura(m: pd.DataFrame, out: str) -> None:
    regimes = [r for r in ROTULO_REGIME if r in set(m.regime)]
    paineis = [("LMC", "final"), ("LMC", "amplitude"), ("SampEn 1D", "final"), ("SampEn 1D", "amplitude")]
    fig, axes = plt.subplots(1, 4, figsize=(15, 5.4), sharey=True, facecolor=SURFACE)
    y = np.arange(len(regimes))[::-1]
    for ax, (est, med) in zip(axes, paineis):
        sub = m[(m.estatistica == est) & (m.medida == med)].set_index("regime").loc[regimes]
        for i, (ds, dy) in enumerate((("mednist", 0.15), ("pathmnist", -0.15))):
            r, p = sub[f"razao_{ds}"].to_numpy(), sub[f"p_{ds}"].to_numpy()
            sig = p < ALFA
            ax.scatter(r[sig], (y + dy)[sig], color=COR[ds], s=34, zorder=3,
                       label=f"{NOME[ds]} (p < 0,05)")
            ax.scatter(r[~sig], (y + dy)[~sig], facecolor=SURFACE, edgecolor=COR[ds], s=34,
                       linewidth=1.4, zorder=3, label=f"{NOME[ds]} (n.s.)")
        ax.axvline(1, color=TEXT_2, linewidth=1, zorder=1)
        ax.set_facecolor(SURFACE)
        ax.grid(True, axis="x", color=GRID, linewidth=0.8)
        for yy in (5.5, 2.5):
            ax.axhline(yy, color=GRID, linewidth=0.8)
        ax.set_title(f"{est} · {med}", color=TEXT_1, fontsize=10.5, loc="left")
        ax.set_xlabel("Razão regime / controle", color=TEXT_2, fontsize=9)
        ax.tick_params(colors=TEXT_2, labelsize=8.5)
        for lado in ("top", "right"):
            ax.spines[lado].set_visible(False)
    axes[0].set_yticks(y, [ROTULO_REGIME[r] for r in regimes])
    alcas, rotulos = axes[0].get_legend_handles_labels()
    fig.legend(alcas, rotulos, frameon=False, fontsize=8.5, labelcolor=TEXT_2,
               loc="lower center", ncol=4)
    fig.suptitle("Efeito de cada regime na camada densa: MedNIST × PathMNIST (ResNet-18)",
                 color=TEXT_1, fontsize=12, x=0.006, ha="left", weight="bold")
    fig.text(0.006, 0.925, "Razão entre a média do regime (n = 5) e a do controle (n = 8). "
             "Ponto cheio: Mann-Whitney p < 0,05, sem correção para múltiplos testes.",
             color=TEXT_2, fontsize=9, ha="left")
    fig.tight_layout(rect=[0, 0.05, 1, 0.9])
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description="Compara os efeitos dos regimes entre MedNIST e PathMNIST")
    p.add_argument("--root", default="monai_weights")
    p.add_argument("--output_dir", default="monai_weights/pm_comparacao_datasets")
    a = p.parse_args()
    os.makedirs(a.output_dir, exist_ok=True)

    m = comparar_regimes()
    dn = densenet_vs_resnet()
    bn = batchnorm_vs_medidas(a.root)
    m.to_csv(os.path.join(a.output_dir, "comparacao_regimes.csv"), index=False)
    dn.to_csv(os.path.join(a.output_dir, "densenet_vs_resnet.csv"), index=False)
    bn.to_csv(os.path.join(a.output_dir, "batchnorm_vs_medidas.csv"), index=False)
    figura(m, os.path.join(a.output_dir, "comparacao_datasets.png"))

    pd.set_option("display.width", 200)
    print(m.round(4).to_string(index=False))
    print("\nPadroes:\n" + m.padrao.value_counts().to_string())
    print("\n" + dn.round(4).to_string(index=False))
    print("\n" + bn.groupby(["dataset", "arquitetura"])[
        ["dp_delta_acc_val", "maior_queda_acc_val", "spearman_lmc", "spearman_sampen"]
    ].mean().round(3).to_string())
    print(f"\nSalvo em {a.output_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


# --- Complexidade: fonte unica de verdade em src/complexity.py (nao redefinir aqui) ---
from src.analysis.overfit_cut import (
    carregar_relatorio,
    marcar_pos_overfit,
    resumo_texto,
    sombrear_regiao_descartada,
)
from src.complexity import (
    find_dense_layers,
    compute_normalized_histogram,
    shannon_entropy_from_hist,
    disequilibrium_from_hist,
    sample_entropy_1d,
    lmc_complexity as _lmc_full,
)


def lmc_complexity(data, bins=100):
    """Valor escalar da LMC. Wrapper fino sobre src.complexity.lmc_complexity.

    bins=100 e mantido EXPLICITO de proposito: o default adaptativo do modulo
    (min(100, max(10, n//5))) divergiria muito em camadas pequenas -- p.ex. um
    BatchNorm de 64 pesos daria 12 bins e uma LMC ~40% diferente.
    """
    return _lmc_full(data, n_bins=bins)["complexity"]


# Nome historico usado por estes scripts; a implementacao canonica e sample_entropy_1d.
sample_entropy = sample_entropy_1d


def compute_metrics_for_weights(weight_path, param_types):
    # --- SOMENTE A CAMADA DENSA (decisao D1 do plan.md) ---
    # Antes, este script percorria TODAS as camadas e subamostrava 10.000 pesos das
    # grandes. A subamostragem foi removida (nenhum peso e descartado), mas a SampEn
    # exata e O(n^2): uma conv de 2,36 M pesos exigiria uma matriz de distancias de
    # 22 TB. Restringir a densa resolve as duas coisas de uma vez -- e e o foco do
    # projeto. Para voltar a analisar conv, seria preciso reintroduzir amostragem,
    # o que invalidaria a SampEn (ela depende da ordem).
    weights = torch.load(weight_path, map_location="cpu", weights_only=True)
    lmc_vals, sampen_vals = [], []
    for name in find_dense_layers(weights, param_types):
        flat = weights[name].detach().cpu().numpy().flatten()
        if flat.size == 0:
            continue
        sample = flat                      # TODOS os pesos, sem amostragem
        lmc = lmc_complexity(sample)
        se = sample_entropy(sample)
        if np.isfinite(lmc) and np.isfinite(se):
            lmc_vals.append(lmc)
            sampen_vals.append(se)
    if not lmc_vals:
        return np.nan, np.nan
    return np.mean(lmc_vals), np.mean(sampen_vals)


def parse_args():
    p = argparse.ArgumentParser(description="Compute LMC + SampEn over epochs for a single run")
    p.add_argument("--run_dir", default="monai_weights/run_mednist_resnet18_64",
                   help="Directory containing epoch_*.pth and param_types.json")
    p.add_argument("--output_dir", default=None,
                   help="Where to write CSV + PNGs (default: <run_dir>/lmc_sampen)")
    p.add_argument("--title_tag", default=None,
                   help="Free-text tag appended to plot titles (default: basename(run_dir))")
    return p.parse_args()


def main():
    args = parse_args()
    run_dir = args.run_dir
    output_dir = args.output_dir or os.path.join(run_dir, "lmc_sampen")
    title_tag = args.title_tag or os.path.basename(os.path.normpath(run_dir))
    os.makedirs(output_dir, exist_ok=True)

    param_types_path = os.path.join(run_dir, "param_types.json")
    with open(param_types_path) as f:
        param_types = json.load(f)

    records = []
    epoch_files = sorted(
        [f for f in os.listdir(run_dir) if f.startswith("epoch_") and f.endswith(".pth")]
    )
    for fname in epoch_files:
        epoch_num = int(fname.replace("epoch_", "").replace(".pth", ""))
        path = os.path.join(run_dir, fname)
        print(f"Processing {fname}...")
        lmc, se = compute_metrics_for_weights(path, param_types)
        records.append({"label": f"Epoch {epoch_num}", "epoch": epoch_num, "LMC": lmc, "SampEn": se})

    df = pd.DataFrame(records).dropna()
    df = df.sort_values("epoch").reset_index(drop=True)

    # Regra D6: as epocas apos o overfitting continuam no grafico (sombreadas), mas
    # ficam marcadas e nao entram em nenhuma conclusao.
    relatorio = carregar_relatorio(run_dir)
    df = marcar_pos_overfit(df, relatorio)
    print(f"Overfitting: {resumo_texto(relatorio)}")
    if relatorio and relatorio.get("confirmado"):
        print(f"  -> {int(df['pos_overfit'].sum())} epoca(s) marcadas como pos-overfitting")

    print(df[["label", "LMC", "SampEn", "pos_overfit"]].to_string())
    df.to_csv(os.path.join(output_dir, "epoch_metrics.csv"), index=False)

    x = df["epoch"]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, df["LMC"], marker="o", color="#1f77b4", linewidth=2, label="LMC Complexity")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean LMC Complexity (across layers)")
    ax.set_title(f"LMC Complexity over Training Epochs — {title_tag}")
    ax.set_xticks(x)
    ax.grid(True, alpha=0.3)
    sombrear_regiao_descartada(ax, relatorio)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "lmc_over_epochs.png"), dpi=150)
    plt.close()
    print("Saved: lmc_over_epochs.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, df["SampEn"], marker="s", color="#d62728", linewidth=2, linestyle="--", label="Sample Entropy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean Sample Entropy (across layers)")
    ax.set_title(f"Sample Entropy over Training Epochs — {title_tag}")
    ax.set_xticks(x)
    ax.grid(True, alpha=0.3)
    sombrear_regiao_descartada(ax, relatorio)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sampen_over_epochs.png"), dpi=150)
    plt.close()
    print("Saved: sampen_over_epochs.png")

    fig, ax1 = plt.subplots(figsize=(10, 5))
    color_lmc = "#1f77b4"
    color_se = "#d62728"

    ax1.plot(x, df["LMC"], marker="o", color=color_lmc, linewidth=2, label="LMC Complexity")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Mean LMC Complexity", color=color_lmc)
    ax1.tick_params(axis="y", labelcolor=color_lmc)
    ax1.set_xticks(x)
    ax1.grid(True, alpha=0.3)
    sombrear_regiao_descartada(ax1, relatorio)

    # NOTA: esta figura usa dois eixos y, que e anti-padrao (duas escalas independentes
    # permitem fazer as curvas se cruzarem onde se quiser). Sera substituida por paineis
    # lado a lado na F6 -- ver plan.md.
    ax2 = ax1.twinx()
    ax2.plot(x, df["SampEn"], marker="s", color=color_se, linewidth=2, linestyle="--", label="Sample Entropy")
    ax2.set_ylabel("Mean Sample Entropy", color=color_se)
    ax2.tick_params(axis="y", labelcolor=color_se)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    plt.title(f"LMC Complexity vs Sample Entropy over Training — {title_tag}\n{resumo_texto(relatorio)}", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "lmc_sampen_combined_epochs.png"), dpi=150)
    plt.close()
    print("Saved: lmc_sampen_combined_epochs.png")


if __name__ == "__main__":
    main()

import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RUN_DIR = "/home/users/u7594034/Área de trabalho/Claude/monai_wieghts/run_mednist_50"
OUTPUT_DIR = "/home/users/u7594034/Área de trabalho/Claude/monai_wieghts/run_mednist_50_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# --- Complexidade: fonte unica de verdade em src/complexity.py (nao redefinir aqui) ---
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


def compute_metrics(weight_path, param_types):
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


with open(os.path.join(RUN_DIR, "param_types.json")) as f:
    param_types = json.load(f)

records = []

epoch_files = sorted(
    [f for f in os.listdir(RUN_DIR) if f.startswith("epoch_") and f.endswith(".pth")]
)
for fname in epoch_files:
    epoch_num = int(fname.replace("epoch_", "").replace(".pth", ""))
    path = os.path.join(RUN_DIR, fname)
    print(f"Processing {fname}...")
    lmc, se = compute_metrics(path, param_types)
    records.append({"label": f"Epoch {epoch_num}", "epoch": epoch_num, "LMC": lmc, "SampEn": se})

df = pd.DataFrame(records).dropna()
df = df.sort_values("epoch").reset_index(drop=True)

print(df[["label", "LMC", "SampEn"]].to_string())
df.to_csv(os.path.join(OUTPUT_DIR, "epoch_metrics.csv"), index=False)

x = df["epoch"]

# --- Plot 1: LMC Complexity ---
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(x, df["LMC"], marker="o", color="#1f77b4", linewidth=2, label="LMC Complexity")
ax.set_xlabel("Epoch")
ax.set_ylabel("Mean LMC Complexity (across layers)")
ax.set_title("LMC Complexity over Training Epochs — MedNIST 50 epochs")
ax.set_xticks(x[::5])
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "lmc_over_epochs.png"), dpi=150)
plt.close()
print("Saved: lmc_over_epochs.png")

# --- Plot 2: Sample Entropy ---
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(x, df["SampEn"], marker="s", color="#d62728", linewidth=2, linestyle="--", label="Sample Entropy")
ax.set_xlabel("Epoch")
ax.set_ylabel("Mean Sample Entropy (across layers)")
ax.set_title("Sample Entropy over Training Epochs — MedNIST 50 epochs")
ax.set_xticks(x[::5])
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "sampen_over_epochs.png"), dpi=150)
plt.close()
print("Saved: sampen_over_epochs.png")

# --- Plot 3: LMC + SampEn dual y-axis ---
fig, ax1 = plt.subplots(figsize=(12, 5))
color_lmc = "#1f77b4"
color_se = "#d62728"

ax1.plot(x, df["LMC"], marker="o", color=color_lmc, linewidth=2, label="LMC Complexity")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Mean LMC Complexity", color=color_lmc)
ax1.tick_params(axis="y", labelcolor=color_lmc)
ax1.set_xticks(x[::5])
ax1.grid(True, alpha=0.3)

ax2 = ax1.twinx()
ax2.plot(x, df["SampEn"], marker="s", color=color_se, linewidth=2, linestyle="--", label="Sample Entropy")
ax2.set_ylabel("Mean Sample Entropy", color=color_se)
ax2.tick_params(axis="y", labelcolor=color_se)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

plt.title("LMC Complexity vs Sample Entropy over Training — MedNIST 50 epochs")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "lmc_sampen_combined_epochs.png"), dpi=150)
plt.close()
print("Saved: lmc_sampen_combined_epochs.png")

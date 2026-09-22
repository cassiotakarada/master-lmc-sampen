import argparse
import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_parser = argparse.ArgumentParser()
_parser.add_argument("--results", type=str, default="results", help="Root directory containing run folders")
_args = _parser.parse_args()

results_root = _args.results
global_output_root = os.path.join(results_root, "global_lmc_sampen")

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

# Organize by dataset
datasets = {}

for folder in os.listdir(results_root):
    if not "_" in folder:
        continue

    run_path = os.path.join(results_root, folder)
    if not os.path.isdir(run_path):
        continue

    for file in os.listdir(run_path):
        if not file.endswith(".pth"):
            continue

        status = "initial" if "initial_" in file else "bef" if "bef_" in file else "aft"
        dataset_key = file.replace("initial_", "").replace("bef_", "").replace("aft_", "").replace(".pth", "")

        weight_path = os.path.join(run_path, file)
        param_path = os.path.join(run_path, f"{file.replace('.pth', '')}_param_types.json")
        if not os.path.exists(param_path):
            param_path = os.path.join(run_path, f"{dataset_key}_param_types.json")
        if not os.path.exists(param_path):
            print(f"⚠️ Missing param_types.json for {file}")
            continue

        datasets.setdefault(dataset_key, {})[status] = {
            "weight": weight_path,
            "param": param_path,
            "folder": folder
        }

# Process each dataset
for dataset, entries in datasets.items():
    print(f"📦 Processing: {dataset}")
    output_folder = os.path.join(global_output_root, dataset)
    os.makedirs(output_folder, exist_ok=True)

    summary = []

    for status in ["initial", "bef", "aft"]:
        if status not in entries:
            continue
        try:
            weights = torch.load(entries[status]["weight"], map_location="cpu")
            with open(entries[status]["param"], "r") as f:
                param_types = json.load(f)
        except Exception as e:
            print(f"❌ Failed loading {dataset} [{status}]: {e}")
            continue

        lmc_vals, sampen_vals, entropy_vals, diseq_vals = [], [], [], []

            # --- SOMENTE A CAMADA DENSA (decisao D1 do plan.md) ---
        # Antes, este script percorria TODAS as camadas e subamostrava 10.000 pesos das
        # grandes. A subamostragem foi removida (nenhum peso e descartado), mas a SampEn
        # exata e O(n^2): uma conv de 2,36 M pesos exigiria uma matriz de distancias de
        # 22 TB. Restringir a densa resolve as duas coisas de uma vez -- e e o foco do
        # projeto. Para voltar a analisar conv, seria preciso reintroduzir amostragem,
        # o que invalidaria a SampEn (ela depende da ordem).
        for name in find_dense_layers(weights, param_types):
            tensor = weights[name]
            flat = tensor.detach().cpu().numpy().flatten()
            # Sem subamostragem: TODOS os pesos do tensor entram no calculo.
            # (A linha antiga sorteava 10.000 pesos, o que embaralhava a ordem e a SampEn
            #  depende da ordem. Removida na F3/F4 -- ver plan.md, problema P4.)
            sample = flat
            hist = compute_normalized_histogram(sample)
            ent = shannon_entropy_from_hist(hist)
            dis = disequilibrium_from_hist(hist)
            lmc = ent * dis
            sampen = sample_entropy(sample)
            if np.isfinite(ent) and np.isfinite(dis) and np.isfinite(lmc) and np.isfinite(sampen):
                entropy_vals.append(ent)
                diseq_vals.append(dis)
                lmc_vals.append(lmc)
                sampen_vals.append(sampen)

        if not lmc_vals: continue
        summary.append({
            "status": status,
            "LMC_mean": np.mean(lmc_vals),
            "Entropy_mean": np.mean(entropy_vals),
            "Desequilibrium_mean": np.mean(diseq_vals),
            "SampEn_mean": np.mean(sampen_vals)
        })

    if not summary:
        print(f"⚠️ No data extracted for {dataset}")
        continue

    df = pd.DataFrame(summary)
    df.to_csv(os.path.join(output_folder, "summary.csv"), index=False)

    stage_order = ["initial", "bef", "aft"]
    df["status"] = pd.Categorical(df["status"], categories=stage_order, ordered=True)
    df = df.sort_values("status")

    # --- plot 1: LMC, Entropy, Disequilibrium (original) ---
    plt.figure(figsize=(8, 6))
    plt.plot(df["status"], df["LMC_mean"], label="Complexidade (LMC)", marker="o", color="cyan")
    plt.plot(df["status"], df["Entropy_mean"], label="Entropia", marker="o", color="red")
    plt.plot(df["status"], df["Desequilibrium_mean"], label="Desequilíbrio", marker="o", color="gold")
    plt.title(f"Complexidade por Estágio - {dataset}")
    plt.xlabel("Estágio do Treinamento")
    plt.ylabel("Valor Médio")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"{dataset}_complexity_plot.png"))
    plt.close()
    print(f"✅ Saved: {dataset}_complexity_plot.png")

    # --- plot 2: LMC + SampEn on dual y-axes ---
    fig, ax1 = plt.subplots(figsize=(8, 5))

    color_lmc = "#1f77b4"
    color_se = "#d62728"

    ax1.plot(df["status"], df["LMC_mean"], marker="o", color=color_lmc, linewidth=2, label="LMC complexity")
    ax1.set_xlabel("Training stage")
    ax1.set_ylabel("LMC complexity (mean across layers)", color=color_lmc)
    ax1.tick_params(axis="y", labelcolor=color_lmc)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(df["status"], df["SampEn_mean"], marker="s", color=color_se, linewidth=2, linestyle="--", label="Sample Entropy")
    ax2.set_ylabel("Sample Entropy (mean across layers)", color=color_se)
    ax2.tick_params(axis="y", labelcolor=color_se)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    plt.title(f"LMC Complexity vs Sample Entropy — {dataset}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"{dataset}_lmc_sampen_plot.png"))
    plt.close()
    print(f"✅ Saved: {dataset}_lmc_sampen_plot.png")

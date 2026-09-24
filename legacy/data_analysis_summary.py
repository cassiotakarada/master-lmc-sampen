import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

results_root = "results"
graph_output_root = os.path.join(results_root, "LMC and SamEn Graphs")
os.makedirs(graph_output_root, exist_ok=True)

palette = {
    "Convolutional": "#1f77b4",
    "Linear": "#2ca02c",
    "Embedding": "#d62728",
    "BatchNorm": "#ff7f0e",
    "Activation": "#9467bd",
    "Pooling": "#8c564b",
    "Flatten": "#e377c2",
    "Other": "#7f7f7f"
}

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

for folder in os.listdir(results_root):
    folder_path = os.path.join(results_root, folder)
    if not os.path.isdir(folder_path) or "_" not in folder:
        continue

    # 🗂️ Extrai número de épocas da pasta (ex: "organmnist3d_64_50" -> 50)
    try:
        epochs = folder.split("_")[-1]
        if not epochs.isdigit():
            epochs = "unknown"
    except:
        epochs = "unknown"

    save_dir = os.path.join(graph_output_root, folder)
    os.makedirs(save_dir, exist_ok=True)

    for file in os.listdir(folder_path):
        if not (file.endswith(".pth") and file.startswith(("aft_", "bef_", "initial_"))):
            continue

        print(f"Processing: {file}")

        weight_path = os.path.join(folder_path, file)
        basename = file.replace(".pth", "")
        param_type_file = f"{basename}_param_types.json"
        param_type_path = os.path.join(folder_path, param_type_file)

        if not os.path.exists(param_type_path):
            print(f"⚠️ Missing param_types.json for {file}")
            continue

        try:
            model_weights = torch.load(weight_path, map_location="cpu")
        except Exception as e:
            print(f"❌ Error loading {file}: {e}")
            continue

        with open(param_type_path, "r") as f:
            param_types = json.load(f)

            records = []
        # --- SOMENTE A CAMADA DENSA (decisao D1 do plan.md) ---
        # Antes, este script percorria TODAS as camadas e subamostrava 10.000 pesos das
        # grandes. A subamostragem foi removida (nenhum peso e descartado), mas a SampEn
        # exata e O(n^2): uma conv de 2,36 M pesos exigiria uma matriz de distancias de
        # 22 TB. Restringir a densa resolve as duas coisas de uma vez -- e e o foco do
        # projeto. Para voltar a analisar conv, seria preciso reintroduzir amostragem,
        # o que invalidaria a SampEn (ela depende da ordem).
        for name in find_dense_layers(model_weights, param_types):
            weight = model_weights[name]
            flat = weight.detach().cpu().numpy().flatten()
            if flat.size == 0:
                continue
            # Sem subamostragem: TODOS os pesos do tensor entram no calculo.
            # (A linha antiga sorteava 10.000 pesos, o que embaralhava a ordem e a SampEn
            #  depende da ordem. Removida na F3/F4 -- ver plan.md, problema P4.)
            sample = flat
            lmc = lmc_complexity(sample)
            sampen = sample_entropy(sample)
            layer_type = param_types.get(name, "Other")
            records.append({
                "layer": name,
                "layer_type": layer_type,
                "LMC": lmc,
                "SampEn": sampen,
                "layer_label": f"{layer_type} | {name}"
            })

        df = pd.DataFrame(records).dropna()
        if df.empty:
            print(f"⚠️ No valid data in {file}")
            continue

        df = df.sort_values(by="layer")

        # 📸 Nome de saída agora inclui número de épocas
        lmc_output = os.path.join(save_dir, f"{basename}_LMC_{epochs}.png")
        sampen_output = os.path.join(save_dir, f"{basename}_SampEn_{epochs}.png")

        if not os.path.exists(lmc_output):
            plt.figure(figsize=(16, 6))
            sns.barplot(x="layer_label", y="LMC", hue="layer_type", data=df, palette=palette, dodge=False)
            plt.xticks(rotation=90)
            plt.title(f"LMC Complexity - {basename} ({epochs} epochs)")
            plt.tight_layout()
            plt.savefig(lmc_output)
            plt.close()
            print(f"✅ Saved: {lmc_output}")

        if not os.path.exists(sampen_output):
            plt.figure(figsize=(16, 6))
            sns.barplot(x="layer_label", y="SampEn", hue="layer_type", data=df, palette=palette, dodge=False)
            plt.xticks(rotation=90)
            plt.title(f"Sample Entropy - {basename} ({epochs} epochs)")
            plt.tight_layout()
            plt.savefig(sampen_output)
            plt.close()
            print(f"✅ Saved: {sampen_output}")

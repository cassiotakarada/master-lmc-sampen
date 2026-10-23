"""
Acerto por classe na VALIDACAO, para todos os treinos -- sem re-treino.

Motivacao: com ruido rosa sigma = 0,2 a acuracia caiu para 83,8% porque UMA classe
inteira (AbdomenCT) passou a ser classificada como ChestCT. A acuracia global esconde
isso; o acerto por classe mostra.

Dois momentos por treino:
  melhor  = bef_weights.pth (epoca de menor val_loss, a mesma avaliada no teste)
  final   = ultimo epoch_XXX.pth (nao existe nos runs de passos igualados)
O conjunto de TESTE nao e usado aqui: ele e olhado uma unica vez, no fim do treino.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pathlib
import re
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
from torch.utils.data import DataLoader

from src.data.mednist import MedNISTDataset, ParticaoMedNIST
from src.data.transforms import build_transforms
from src.training.model import build_model

SURFACE, TEXT_1, TEXT_2 = "#fcfcfb", "#0b0b0b", "#52514e"

# (regime, padrao) -- mesma nomenclatura de scripts/comportamento_densa.py
REGIMES = [
    ("controle", "f6_ctrlpar_seed*"),
    ("frac10", "f6_frac10_seed*"), ("frac02", "f6_frac02_seed*"), ("frac01", "f6_frac01_seed*"),
    ("frac10eq", "f6_frac10eq_seed*"), ("frac02eq", "f6_frac02eq_seed*"), ("frac01eq", "f6_frac01eq_seed*"),
    ("rosa005", "f9_rosa_s005_seed*"), ("rosa010", "f9_rosa_s010_seed*"), ("rosa020", "f9_rosa_s020_seed*"),
    ("branco005", "f9_branco_s005_seed*"), ("branco010", "f9_branco_s010_seed*"),
    ("branco020", "f9_branco_s020_seed*"),
    ("dn_controle", "dn121_ctrlpar_seed*"),
]


def carregar_pesos(modelo, caminho):
    sd = torch.load(caminho, map_location="cpu", weights_only=True)
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    modelo.load_state_dict(sd)


def avaliar(modelo, loader, dev, n_classes):
    acertos = np.zeros(n_classes)
    totais = np.zeros(n_classes)
    modelo.eval()
    with torch.no_grad():
        for b in loader:
            pred = modelo(b["image"].to(dev)).argmax(1).cpu().numpy()
            y = b["label"].numpy()
            for c in range(n_classes):
                sel = y == c
                totais[c] += sel.sum()
                acertos[c] += (pred[sel] == c).sum()
    return acertos / np.maximum(totais, 1)


def main() -> int:
    p = argparse.ArgumentParser(description="Acerto por classe na validacao")
    p.add_argument("--root", default="monai_weights")
    p.add_argument("--split_file", default="data/mednist_split.json")
    p.add_argument("--output_dir", default="monai_weights/f8_acerto_por_classe")
    a = p.parse_args()
    os.makedirs(a.output_dir, exist_ok=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    particao = ParticaoMedNIST(a.split_file)
    classes = particao.classes
    val = MedNISTDataset("val", transform=build_transforms((64, 64)), particao=particao)
    loader = DataLoader(val, batch_size=512, num_workers=4)

    linhas = []
    for regime, padrao in REGIMES:
        for d in sorted(x for x in glob.glob(os.path.join(a.root, padrao)) if os.path.isdir(x)):
            if not os.path.isfile(os.path.join(d, "test_metrics.json")):
                continue                                   # treino ainda em andamento
            cfg = json.load(open(os.path.join(d, "config.json")))
            cks = sorted(glob.glob(os.path.join(d, "epoch_*.pth")),
                         key=lambda q: int(re.search(r"epoch_(\d+)", q).group(1)))
            momentos = [("melhor", os.path.join(d, "bef_weights.pth"),
                         json.load(open(os.path.join(d, "test_metrics.json")))["epoca_avaliada"])]
            if cks:
                momentos.append(("final", cks[-1], int(re.search(r"epoch_(\d+)", cks[-1]).group(1))))
            modelo = build_model(cfg["model_name"], 3, len(classes)).to(dev)
            for momento, caminho, epoca in momentos:
                carregar_pesos(modelo, caminho)
                acc = avaliar(modelo, loader, dev, len(classes))
                for c, nome in enumerate(classes):
                    linhas.append({"regime": regime, "run": os.path.basename(d), "semente": cfg["seed"],
                                   "momento": momento, "epoca": epoca, "classe": nome,
                                   "acerto": float(acc[c])})
            print(f"{os.path.basename(d)}: ok", flush=True)

    df = pd.DataFrame(linhas)
    df.to_csv(os.path.join(a.output_dir, "acerto_por_classe.csv"), index=False)

    # resumo: media entre sementes, por regime x classe, no momento final (ou melhor, se
    # nao houver final)
    def momento_preferido(g):
        return g[g.momento == "final"] if (g.momento == "final").any() else g[g.momento == "melhor"]
    pref = df.groupby("run", group_keys=False).apply(momento_preferido)
    ordem = [r for r, _ in REGIMES if r in set(pref.regime)]
    tab = pref.pivot_table(index="regime", columns="classe", values="acerto", aggfunc="mean")
    tab = tab.loc[ordem, classes]
    tab["minimo"] = tab.min(axis=1)
    tab.to_csv(os.path.join(a.output_dir, "acerto_por_classe_resumo.csv"))

    fig, ax = plt.subplots(figsize=(10, 0.45 * len(tab) + 1.6), facecolor=SURFACE)
    vals = tab[classes].to_numpy()
    im = ax.imshow(vals, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            ax.text(j, i, f"{vals[i, j] * 100:.1f}", ha="center", va="center", fontsize=8.5,
                    color="#ffffff" if vals[i, j] > 0.6 else TEXT_1)
    ax.set_xticks(range(len(classes)), classes, fontsize=9, color=TEXT_2)
    ax.set_yticks(range(len(tab)), tab.index, fontsize=9, color=TEXT_2)
    ax.set_title("Acerto por classe na validação (%), média entre sementes — última época "
                 "(ou melhor época, nos runs sem checkpoint final)", color=TEXT_1, fontsize=10.5, loc="left")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(os.path.join(a.output_dir, "acerto_por_classe.png"), dpi=150, facecolor=SURFACE)
    pd.set_option("display.width", 200)
    print((tab * 100).round(1).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())

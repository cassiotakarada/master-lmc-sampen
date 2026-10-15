"""
Figura de exemplo do ruido nas imagens: a mesma imagem do MedNIST limpa, com ruido
branco, rosa e marrom, os campos de ruido e o espectro de potencia radial de cada um.

Uso: python scripts/exemplo_ruido_imagem.py [--sigma 0.1] [--idx 0]
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import fft

from src.data.image_noise import CORES, RuidoNaImagem
from src.data.mednist import MedNISTDataset, ParticaoMedNIST
from src.data.transforms import build_transforms

SURFACE, TEXT_1, TEXT_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2de"
COR_LINHA = {"branco": "#7c7b76", "rosa": "#e87ba4", "marrom": "#a83a12"}


def espectro_radial(campo: np.ndarray, n_faixas: int = 30):
    pot = np.abs(fft.fft2(campo)) ** 2
    h, w = campo.shape
    f = np.sqrt(fft.fftfreq(h)[:, None] ** 2 + fft.fftfreq(w)[None, :] ** 2)
    faixas = np.geomspace(1.0 / max(h, w), 0.5, n_faixas + 1)
    centro, media = [], []
    for a, b in zip(faixas[:-1], faixas[1:]):
        sel = (f >= a) & (f < b)
        if sel.any():
            centro.append(np.sqrt(a * b))
            media.append(pot[sel].mean())
    return np.array(centro), np.array(media)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--sigma", type=float, default=0.1)
    p.add_argument("--idx", type=int, default=0)
    p.add_argument("--split_file", default="data/mednist_split.json")
    p.add_argument("--out", default="monai_weights/f9_ruido_imagem/exemplo_ruido.png")
    a = p.parse_args()

    base = MedNISTDataset("train", transform=build_transforms((64, 64)),
                          particao=ParticaoMedNIST(a.split_file))
    limpa = base[a.idx]["image"][0].numpy()

    fig, axes = plt.subplots(3, 4, figsize=(15, 11), facecolor=SURFACE,
                             gridspec_kw={"height_ratios": [1, 1, 0.9]})
    axes[0, 0].imshow(limpa, cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_title("Imagem original", color=TEXT_1, fontsize=11, loc="left")
    axes[1, 0].axis("off")
    axes[1, 0].text(0.0, 0.5, f"Mesma imagem do MedNIST (treino, índice {a.idx}).\n"
                    f"σ = {a.sigma} na escala do pixel [0, 1].\n\nLinha de cima: imagem + ruído.\n"
                    "Linha do meio: só o campo de ruído.\nEmbaixo: espectro de potência radial.",
                    color=TEXT_2, fontsize=10, va="center")
    for j, cor in enumerate(("branco", "rosa", "marrom"), start=1):
        ds = RuidoNaImagem(base, beta=CORES[cor], sigma=a.sigma, seed=42)
        ruidosa = ds[a.idx]["image"][0].numpy()
        campo = ds.ruido(a.idx, 64, 64)
        axes[0, j].imshow(ruidosa, cmap="gray", vmin=0, vmax=1)
        axes[0, j].set_title(f"+ ruído {cor} (β = {CORES[cor]:.0f})", color=TEXT_1, fontsize=11, loc="left")
        axes[1, j].imshow(campo, cmap="gray")
        axes[1, j].set_title(f"campo de ruído {cor}", color=TEXT_1, fontsize=11, loc="left")
    for ax in axes[:2].ravel():
        ax.set_xticks([]); ax.set_yticks([])

    gs = axes[2, 0].get_gridspec()
    for ax in axes[2]:
        ax.remove()
    ax = fig.add_subplot(gs[2, :])
    rng = np.random.default_rng(0)
    for cor in ("branco", "rosa", "marrom"):
        acum = None
        for _ in range(50):
            from src.data.image_noise import ruido_colorido
            fx, pw = espectro_radial(ruido_colorido(64, 64, CORES[cor], rng))
            acum = pw if acum is None else acum + pw
        ax.loglog(fx, acum / 50, color=COR_LINHA[cor], linewidth=2.2,
                  label=f"{cor}: potência ∝ 1/f^{CORES[cor]:.0f}")
    ax.set_facecolor(SURFACE)
    ax.set_xlabel("Frequência espacial radial f (ciclos/pixel, log)", color=TEXT_2, fontsize=10)
    ax.set_ylabel("Potência média (log)", color=TEXT_2, fontsize=10)
    ax.set_title("Espectro de potência radial (média de 50 sorteios)", color=TEXT_1, fontsize=11, loc="left")
    ax.grid(True, color=GRID, linewidth=0.8, which="both")
    ax.tick_params(colors=TEXT_2)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    ax.legend(frameon=False, fontsize=10, labelcolor=TEXT_2)

    fig.suptitle("Ruído colorido nas imagens de treino — branco, rosa e marrom",
                 color=TEXT_1, fontsize=13, x=0.006, ha="left", weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=140, facecolor=SURFACE)
    print(a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

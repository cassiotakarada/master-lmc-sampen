"""SampEn 2D analysis on final checkpoints — focused on conv1 kernel mosaics.

Why only conv1: ResNet-18's other Conv2d layers use 3×3 kernels, which are too
small for SampEn 2D windows of size m+1=3 (only one valid window position).
The conv1 layer uses 7×7 kernels — large enough for several (3×3) window
positions when arranged as a mosaic.

Reshape convention: for each input channel c ∈ {0,1,2}, the 64 output filters
(each 7×7) are tiled into an 8×8 grid → a single 56×56 image. SampEn 2D is
computed on this image with r=0.5σ (calibrated empirically — the default
r=0.2σ produces all-inf because trained weight mosaics have low local
regularity).

This isolates the *spatial regularity of conv1 filter arrangements* as a
single scalar per (dataset, seed, input-channel). We aggregate across the 3
input channels (which, after the RGB-replicate transform of make_loaders, are
independently trained even when the inputs are identical) and the 3 seeds.

Outputs (results/sampen2d_figs/):
    s2d_f1_conv1_mosaic.{pdf,png}    — bar chart per dataset with 95% CI
    s2d_f2_per_channel.{pdf,png}     — three bars per dataset (one per input channel)
    s2d_table.{csv,tex}              — quantitative summary
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.complexity import sample_entropy_2d  # noqa: E402

THESIS_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR    = THESIS_ROOT / "results" / "runs"
OUT         = THESIS_ROOT / "results" / "sampen2d_figs"
OUT.mkdir(parents=True, exist_ok=True)

DATASETS = ["PathMNIST", "OrganAMNIST", "BloodMNIST"]
SEEDS    = [42, 123, 2024]
EPOCHS   = 100
R_SCALE  = 0.5     # empirically calibrated — 0.2 produces all-inf

DS_COLOR = {
    "PathMNIST":   "#1f77b4",
    "OrganAMNIST": "#2ca02c",
    "BloodMNIST":  "#d62728",
}

plt.rcParams.update({
    "font.size": 10.5, "axes.titlesize": 11.5, "axes.labelsize": 10.5,
    "legend.fontsize": 9, "figure.dpi": 130,
})


def load_conv1(ds, seed):
    p = RUNS_DIR / f"{ds}_{EPOCHS}ep_seed{seed}" / "checkpoints" / f"resume_epoch_{EPOCHS}.pt"
    if not p.exists():
        return None
    ckpt = torch.load(p, map_location="cpu", weights_only=False)
    sd = ckpt["model"]
    return sd["conv1.weight"].numpy()  # [64, 3, 7, 7]


def mosaic_for_channel(conv1_w: np.ndarray, c: int) -> np.ndarray:
    """Build 56×56 mosaic: 8×8 grid of 7×7 filters for input channel c."""
    filters_c = conv1_w[:, c, :, :]            # [64, 7, 7]
    if filters_c.shape[0] != 64 or filters_c.shape[1] != 7 or filters_c.shape[2] != 7:
        raise ValueError(f"Unexpected conv1 shape: {conv1_w.shape}")
    # Reshape (64,7,7) → (8,8,7,7) → transpose → (8,7,8,7) → reshape (56,56)
    grid = filters_c.reshape(8, 8, 7, 7).transpose(0, 2, 1, 3).reshape(56, 56)
    return grid


def collect():
    rows = []
    for ds in DATASETS:
        for seed in SEEDS:
            conv1 = load_conv1(ds, seed)
            if conv1 is None:
                print(f"missing: {ds} seed={seed}")
                continue
            for c in range(3):
                mosaic = mosaic_for_channel(conv1, c)
                s = sample_entropy_2d(mosaic, m=2, r_scale=R_SCALE)
                rows.append({
                    "dataset":      ds,
                    "seed":         seed,
                    "input_channel": c,
                    "sampen_2d":    float(s) if np.isfinite(s) else np.nan,
                })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "s2d_raw_conv1.csv", index=False)
    return df


def fig_bar_with_ci(df):
    """Bar chart: mean ± 95% bootstrap CI per dataset (over 9 = 3 channels × 3 seeds)."""
    rng = np.random.default_rng(0)
    means, lo, hi = [], [], []
    for ds in DATASETS:
        vals = df[df["dataset"] == ds]["sampen_2d"].dropna().values
        if len(vals) == 0:
            means.append(np.nan); lo.append(np.nan); hi.append(np.nan); continue
        mean = float(np.mean(vals))
        # Bootstrap CI over the values (no weighting — 9 points)
        boot = [np.mean(rng.choice(vals, len(vals), replace=True)) for _ in range(1000)]
        means.append(mean)
        lo.append(float(np.percentile(boot, 2.5)))
        hi.append(float(np.percentile(boot, 97.5)))

    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    x = np.arange(len(DATASETS))
    err = np.vstack([np.array(means) - np.array(lo),
                     np.array(hi) - np.array(means)])
    ax.bar(x, means, color=[DS_COLOR[d] for d in DATASETS],
           alpha=0.78, edgecolor="black", linewidth=0.8)
    ax.errorbar(x, means, yerr=err, fmt="none", ecolor="black",
                capsize=6, linewidth=1.4)
    ax.set_xticks(x); ax.set_xticklabels(DATASETS, fontsize=10)
    ax.set_xlabel("Dataset")
    ax.set_ylabel(r"SampEn$_{2D}$ — mosaico 56$\times$56 dos filtros da conv1")
    ax.set_title("SampEn$_{2D}$ no checkpoint final (3 sementes × 3 canais; 95% CI bootstrap)")
    ax.grid(alpha=0.3, axis="y")
    for i, m in enumerate(means):
        if np.isfinite(m):
            ax.text(i, hi[i] + 0.06, f"{m:.2f}", ha="center", va="bottom",
                    fontsize=10, weight="bold")
    plt.tight_layout()
    plt.savefig(OUT / "s2d_f1_conv1_mosaic.pdf"); plt.savefig(OUT / "s2d_f1_conv1_mosaic.png")
    plt.close()
    print("S2D-F1 ok")
    return pd.DataFrame({"dataset": DATASETS, "mean": means, "ci_low": lo, "ci_high": hi})


def fig_per_channel(df):
    """3 bars per dataset — one for each input channel — to see if grayscale (OrganAMNIST) shows channel-uniformity."""
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    width = 0.25
    x = np.arange(len(DATASETS))
    for i, c in enumerate([0, 1, 2]):
        means = []
        stds = []
        for ds in DATASETS:
            vals = df[(df["dataset"] == ds) & (df["input_channel"] == c)]["sampen_2d"].dropna().values
            means.append(np.mean(vals) if len(vals) else np.nan)
            stds.append(np.std(vals) if len(vals) > 1 else 0.0)
        ax.bar(x + (i - 1) * width, means, width,
               yerr=stds, capsize=3,
               label=f"canal {c}", alpha=0.85, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels(DATASETS)
    ax.set_xlabel("Dataset")
    ax.set_ylabel(r"SampEn$_{2D}$ (mosaico 56$\times$56)")
    ax.set_title("SampEn$_{2D}$ por canal de entrada da conv1 (médias entre 3 sementes; barras: $\pm 1\\sigma$)")
    ax.grid(alpha=0.3, axis="y"); ax.legend()
    plt.tight_layout()
    plt.savefig(OUT / "s2d_f2_per_channel.pdf"); plt.savefig(OUT / "s2d_f2_per_channel.png")
    plt.close()
    print("S2D-F2 ok")


def write_table(summary):
    summary.to_csv(OUT / "s2d_table.csv", index=False)
    lines = [
        "% Auto-generated by scripts/make_sampen2d_figs.py",
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{$\\mathrm{SampEn}_{2D}$ sobre mosaicos $56\\times 56$ dos "
        "filtros 7$\\times$7 da \\texttt{conv1} no \\emph{checkpoint} final "
        "(épocas: 100; mosaicos: 8$\\times$8 dos 64 filtros; tolerância "
        f"$r = {R_SCALE}\\,\\sigma$; média entre 3 sementes $\\times$ 3 "
        "canais de entrada; intervalo de confiança 95\\% por \\emph{bootstrap}).}",
        "\\label{tab:sampen2d-conv1}",
        "\\small",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Dataset & média $\\mathrm{SampEn}_{2D}$ & IC $95\\%$ inferior & superior \\\\",
        "\\midrule",
    ]
    for _, r in summary.iterrows():
        lines.append(
            f"{r['dataset']} & {r['mean']:.3f} & {r['ci_low']:.3f} & {r['ci_high']:.3f} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (OUT / "s2d_table.tex").write_text("\n".join(lines) + "\n")
    print("S2D-table ok")


def main():
    df = collect()
    print(f"\nCollected {len(df)} (dataset, seed, channel) data points")
    print(df.to_string(index=False))
    summary = fig_bar_with_ci(df)
    fig_per_channel(df)
    write_table(summary)
    print("\nFinal summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

"""Aggregate LMC + SampEn trajectories across multiple runs into one figure.

Reuses each run's <run_dir>/lmc_sampen/epoch_metrics.csv if present; otherwise
recomputes from epoch_*.pth checkpoints. Useful for seed-sweep comparison.
"""
import argparse
import json
import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from monai_lmc_sampen_plot import compute_metrics_for_weights
from src.analysis.overfit_cut import carregar_relatorio, marcar_pos_overfit


def load_or_compute_metrics(run_dir: str, csv_subdir: str = "lmc_sampen") -> pd.DataFrame:
    csv_path = os.path.join(run_dir, csv_subdir, "epoch_metrics.csv")
    if os.path.isfile(csv_path):
        df = pd.read_csv(csv_path)
        return df.dropna().sort_values("epoch").reset_index(drop=True)

    print(f"[compute] no cached metrics for {run_dir}, computing from checkpoints")
    pt_path = os.path.join(run_dir, "param_types.json")
    with open(pt_path) as f:
        param_types = json.load(f)

    records = []
    epoch_files = sorted(
        f for f in os.listdir(run_dir) if f.startswith("epoch_") and f.endswith(".pth")
    )
    for fname in epoch_files:
        epoch = int(fname.replace("epoch_", "").replace(".pth", ""))
        lmc, se = compute_metrics_for_weights(os.path.join(run_dir, fname), param_types)
        records.append({"epoch": epoch, "LMC": lmc, "SampEn": se})

    df = pd.DataFrame(records).dropna().sort_values("epoch").reset_index(drop=True)
    df = marcar_pos_overfit(df, carregar_relatorio(run_dir))

    out_dir = os.path.join(run_dir, csv_subdir)
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(csv_path, index=False)
    return df


def stack_runs(run_dirs: List[str]) -> Tuple[pd.DataFrame, dict]:
    per_run = {}
    for rd in run_dirs:
        df = load_or_compute_metrics(rd)
        per_run[rd] = df

    all_epochs = sorted({int(e) for df in per_run.values() for e in df["epoch"]})
    rows = []
    for epoch in all_epochs:
        lmc_vals, se_vals = [], []
        for df in per_run.values():
            row = df[df["epoch"] == epoch]
            if not row.empty:
                lmc_vals.append(float(row["LMC"].iloc[0]))
                se_vals.append(float(row["SampEn"].iloc[0]))
        if lmc_vals:
            rows.append({
                "epoch": epoch,
                "LMC_mean": float(np.mean(lmc_vals)),
                "LMC_std": float(np.std(lmc_vals, ddof=0)),
                "SampEn_mean": float(np.mean(se_vals)),
                "SampEn_std": float(np.std(se_vals, ddof=0)),
                "n_runs": len(lmc_vals),
            })
    return pd.DataFrame(rows), per_run


def plot(agg: pd.DataFrame, per_run: dict, output_dir: str, title_tag: str):
    os.makedirs(output_dir, exist_ok=True)
    x = agg["epoch"].values

    # LMC: per-run thin lines + mean±std band
    fig, ax = plt.subplots(figsize=(10, 5))
    for rd, df in per_run.items():
        ax.plot(df["epoch"], df["LMC"], color="#1f77b4", alpha=0.25, linewidth=1,
                label=os.path.basename(rd.rstrip("/")))
    ax.plot(x, agg["LMC_mean"], color="#1f77b4", linewidth=2.5, marker="o", label="mean")
    ax.fill_between(x, agg["LMC_mean"] - agg["LMC_std"], agg["LMC_mean"] + agg["LMC_std"],
                    color="#1f77b4", alpha=0.2, label="±1σ")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean LMC Complexity (across layers)")
    ax.set_title(f"LMC Complexity over Epochs — {title_tag} (n={len(per_run)} runs)")
    ax.set_xticks(x)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "lmc_aggregate.png"), dpi=150)
    plt.close()

    # SampEn
    fig, ax = plt.subplots(figsize=(10, 5))
    for rd, df in per_run.items():
        ax.plot(df["epoch"], df["SampEn"], color="#d62728", alpha=0.25, linewidth=1,
                linestyle="--", label=os.path.basename(rd.rstrip("/")))
    ax.plot(x, agg["SampEn_mean"], color="#d62728", linewidth=2.5, marker="s", label="mean")
    ax.fill_between(x, agg["SampEn_mean"] - agg["SampEn_std"], agg["SampEn_mean"] + agg["SampEn_std"],
                    color="#d62728", alpha=0.2, label="±1σ")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean Sample Entropy (across layers)")
    ax.set_title(f"Sample Entropy over Epochs — {title_tag} (n={len(per_run)} runs)")
    ax.set_xticks(x)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sampen_aggregate.png"), dpi=150)
    plt.close()

    # Combined dual-axis means
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(x, agg["LMC_mean"], color="#1f77b4", linewidth=2, marker="o", label="LMC mean")
    ax1.fill_between(x, agg["LMC_mean"] - agg["LMC_std"], agg["LMC_mean"] + agg["LMC_std"],
                     color="#1f77b4", alpha=0.2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Mean LMC Complexity", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.set_xticks(x)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(x, agg["SampEn_mean"], color="#d62728", linewidth=2, marker="s",
             linestyle="--", label="SampEn mean")
    ax2.fill_between(x, agg["SampEn_mean"] - agg["SampEn_std"], agg["SampEn_mean"] + agg["SampEn_std"],
                     color="#d62728", alpha=0.2)
    ax2.set_ylabel("Mean Sample Entropy", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    plt.title(f"LMC vs SampEn (mean ± 1σ) — {title_tag} (n={len(per_run)} runs)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "lmc_sampen_aggregate.png"), dpi=150)
    plt.close()

    agg.to_csv(os.path.join(output_dir, "aggregate_metrics.csv"), index=False)
    print(f"Wrote 3 PNGs + aggregate_metrics.csv to {output_dir}")


def parse_args():
    p = argparse.ArgumentParser(description="Aggregate LMC/SampEn across runs")
    p.add_argument("--run_dirs", nargs="+", required=True,
                   help="Two or more run directories to aggregate")
    p.add_argument("--output_dir", required=True,
                   help="Where to write aggregate plots/CSV")
    p.add_argument("--title_tag", default="seed sweep",
                   help="Title tag for plots")
    return p.parse_args()


def main():
    args = parse_args()
    agg, per_run = stack_runs(args.run_dirs)
    plot(agg, per_run, args.output_dir, args.title_tag)


if __name__ == "__main__":
    main()

"""Cross-dataset analysis of the LMC peak invariance.

Uses the 27 baseline runs (no controlled-experiment overlays — just the
plain (dataset × epochs × seed) grid from the original experiment_grid.yaml):
    {PathMNIST, OrganAMNIST, BloodMNIST} × {30, 50, 100} ep × {42, 123, 2024}.

Goal: test whether the LMC peak window ~[4, 13] epochs established for
PathMNIST in Phases 1-3 holds for OrganAMNIST and BloodMNIST.

Outputs (results/crossds_figs/):
    cds_f1_lmc_trajectories.{pdf,png}   — LMC × epoch, 3 datasets, 100ep runs
    cds_f2_lead_time.{pdf,png}          — lead-time bar chart with bootstrap CI
    cds_f3_inflection_table.csv/.tex    — quantitative summary table
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from robust_detector import bootstrap_peak_ci  # noqa: E402

THESIS_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR    = THESIS_ROOT / "results" / "runs"
OUT         = THESIS_ROOT / "results" / "crossds_figs"
OUT.mkdir(parents=True, exist_ok=True)

DATASETS = ["PathMNIST", "OrganAMNIST", "BloodMNIST"]
SEEDS    = [42, 123, 2024]
EPOCHS   = 100   # use 100-ep runs only for direct comparability with Phases 1-3

DS_COLOR = {
    "PathMNIST":   "#1f77b4",
    "OrganAMNIST": "#2ca02c",
    "BloodMNIST":  "#d62728",
}
DS_LABEL = {
    "PathMNIST":   "PathMNIST (9 classes, RGB histología)",
    "OrganAMNIST": "OrganAMNIST (11 classes, CT cinza)",
    "BloodMNIST":  "BloodMNIST (8 classes, RGB microscopia)",
}

# Robust detector convention — identical to Phases 1-3
ROBUST_GROUP   = "bn_gamma"
ROBUST_MIN_EP  = 0.5
ROBUST_WINDOW  = 7
ROBUST_POLY    = 3
ROBUST_NBOOT   = 1000
ROBUST_CI      = 0.95

plt.rcParams.update({
    "font.size": 10.5, "axes.titlesize": 11.5, "axes.labelsize": 10.5,
    "legend.fontsize": 9, "figure.dpi": 130,
})


def load_history(ds, seed):
    p = RUNS_DIR / f"{ds}_{EPOCHS}ep_seed{seed}" / "history.csv"
    return pd.read_csv(p) if p.exists() else None


def load_metrics(ds, seed):
    p = RUNS_DIR / f"{ds}_{EPOCHS}ep_seed{seed}" / "metrics.csv"
    return pd.read_csv(p) if p.exists() else None


def min_val_loss_epoch(history):
    val = history[history["phase"] == "val"]
    return float(val.loc[val["loss"].idxmin(), "epoch"]) if len(val) else float("nan")


def aggregate_metric_over_seeds(ds, metric_col, group=None):
    dfs = []
    for seed in SEEDS:
        df = load_metrics(ds, seed)
        if df is None:
            continue
        if group is not None:
            df = df[df["group"] == group]
        else:
            df = df.groupby("step", as_index=False).agg(
                epoch=("epoch", "first"),
                **{metric_col: (metric_col, "mean")},
            )
        df = df[["step", "epoch", metric_col]].copy()
        df["seed"] = seed
        dfs.append(df)
    if not dfs:
        return None
    cat = pd.concat(dfs, ignore_index=True)
    return cat.groupby("step").agg(
        epoch=("epoch", "first"),
        mean=(metric_col, "mean"),
        std=(metric_col, "std"),
    ).reset_index().sort_values("step")


def _bn_gamma_late_dfs(ds):
    dfs = []
    for seed in SEEDS:
        m = load_metrics(ds, seed)
        if m is None:
            continue
        sub = m[(m["group"] == ROBUST_GROUP) & (m["epoch"] >= ROBUST_MIN_EP)]
        sub = sub[["step", "epoch", "lmc_complexity"]].copy()
        if not sub.empty:
            dfs.append(sub)
    return dfs


# ============================================================================
def fig_lmc_trajectories():
    fig, ax = plt.subplots(figsize=(10, 5))
    for ds in DATASETS:
        agg = aggregate_metric_over_seeds(ds, "lmc_complexity")
        if agg is None or len(agg) == 0:
            continue
        ax.plot(agg["epoch"], agg["mean"], color=DS_COLOR[ds],
                linewidth=2, label=DS_LABEL[ds])
        ax.fill_between(agg["epoch"], agg["mean"] - agg["std"], agg["mean"] + agg["std"],
                        color=DS_COLOR[ds], alpha=0.16)
        e_stars = [min_val_loss_epoch(load_history(ds, s)) for s in SEEDS
                   if load_history(ds, s) is not None]
        if e_stars:
            ax.axvline(float(np.nanmean(e_stars)), color=DS_COLOR[ds], ls=":",
                       lw=1.0, alpha=0.7)
    ax.set_xscale("symlog", linthresh=1e-2)
    ax.set_xlabel("Época (simlog)")
    ax.set_ylabel("LMC complexity (média entre camadas)")
    ax.set_title("LMC × época por dataset (100 épocas, baseline; 3 sementes)")
    ax.grid(alpha=0.3); ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT / "cds_f1_lmc_trajectories.pdf"); plt.savefig(OUT / "cds_f1_lmc_trajectories.png")
    plt.close()
    print("CDS-F1 ok")


def fig_lead_time():
    means, lo, hi, labels, colors = [], [], [], [], []
    rows = []
    for ds in DATASETS:
        e_val_list = []
        for seed in SEEDS:
            h = load_history(ds, seed)
            if h is None:
                continue
            val = h[h["phase"] == "val"]
            e_val_list.append(float(val.loc[val["loss"].idxmin(), "epoch"]))
        if not e_val_list:
            continue
        e_val_mean = float(np.nanmean(e_val_list))

        dfs = _bn_gamma_late_dfs(ds)
        if not dfs:
            continue
        peak = bootstrap_peak_ci(dfs, "lmc_complexity",
                                 n_bootstrap=ROBUST_NBOOT, ci_level=ROBUST_CI,
                                 window_size=ROBUST_WINDOW, polyorder=ROBUST_POLY)

        labels.append(ds)
        colors.append(DS_COLOR[ds])
        means.append(e_val_mean - peak.epoch_peak)
        lo.append(e_val_mean - peak.ci_high)
        hi.append(e_val_mean - peak.ci_low)

        # val acc/loss for table
        vmin_list, vacc_list = [], []
        for seed in SEEDS:
            h = load_history(ds, seed)
            if h is None:
                continue
            val = h[h["phase"] == "val"]
            vmin_list.append(float(val["loss"].min()))
            vacc_list.append(float(val.loc[val["loss"].idxmin(), "acc"]))

        rows.append({
            "dataset":            ds,
            "epoch_min_val_loss": e_val_mean,
            "min_val_loss":       float(np.nanmean(vmin_list)),
            "val_acc_at_min":     float(np.nanmean(vacc_list)),
            "epoch_peak_lmc":     peak.epoch_peak,
            "lmc_ci_low":         peak.ci_low,
            "lmc_ci_high":        peak.ci_high,
            "lead_time_lmc":      e_val_mean - peak.epoch_peak,
        })

    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    x = np.arange(len(labels))
    err = np.vstack([np.array(means) - np.array(lo),
                     np.array(hi) - np.array(means)])
    ax.bar(x, means, color=colors, alpha=0.78, edgecolor="black", linewidth=0.8)
    ax.errorbar(x, means, yerr=err, fmt="none", ecolor="black", capsize=6, linewidth=1.4)
    ax.axhline(0, color="gray", linewidth=0.7)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_xlabel("Dataset")
    ax.set_ylabel(r"Lead time = $e^*_\mathrm{val} - e^{\mathrm{peak}}_\mathrm{LMC}$ (épocas)")
    ax.set_title("Lead time da LMC por dataset (bn$_\\gamma$, epoch $\\geq 0.5$) — 95% CI bootstrap")
    ax.grid(alpha=0.3, axis="y")
    for i, (m, h) in enumerate(zip(means, hi)):
        ax.text(i, max(h, m) + 0.4, f"{m:+.1f}",
                ha="center", va="bottom", fontsize=10, weight="bold")
    plt.tight_layout()
    plt.savefig(OUT / "cds_f2_lead_time.pdf"); plt.savefig(OUT / "cds_f2_lead_time.png")
    plt.close()
    print("CDS-F2 ok")
    return pd.DataFrame(rows)


def write_table(df):
    df.to_csv(OUT / "cds_f3_inflection_table.csv", index=False)
    lines = [
        "% Auto-generated by scripts/make_crossds_figs.py",
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{Pontos de inflexão e \\emph{lead time} entre datasets MedMNIST "
        "(PathMNIST, OrganAMNIST, BloodMNIST; 100 épocas, 3 sementes; \\textit{bn\\_gamma}; "
        "bootstrap 95\\% CI).}",
        "\\label{tab:crossds-inflexao}",
        "\\small",
        "\\begin{tabular}{lrcrr}",
        "\\toprule",
        "Dataset & $e^*_\\mathrm{val}$ & min val\\_loss & $e^{\\text{peak}}_\\mathrm{LMC}$ "
        "[95\\% CI] & lead time \\\\",
        "\\midrule",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"{r['dataset']} & "
            f"{r['epoch_min_val_loss']:.1f} & "
            f"{r['min_val_loss']:.3f} & "
            f"{r['epoch_peak_lmc']:.1f} [{r['lmc_ci_low']:.1f}, {r['lmc_ci_high']:.1f}] & "
            f"{r['lead_time_lmc']:+.1f} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (OUT / "cds_f3_inflection_table.tex").write_text("\n".join(lines) + "\n")
    print("CDS-F3 ok")
    print(df.to_string(index=False))


def main():
    fig_lmc_trajectories()
    df = fig_lead_time()
    write_table(df)
    print(f"\nAll figures written to: {OUT}")


if __name__ == "__main__":
    main()

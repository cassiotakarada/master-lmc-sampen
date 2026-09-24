"""Phase-2 figures: weight-decay sweep × LMC/SampEn trajectories.

Reads the 9 runs of config/experiment_grid_wd.yaml plus the 3 baseline runs
(PathMNIST_100ep_seed* with WD=1e-4) from the original grid, and produces:
    wd_f1_val_curves.{pdf,png}    — train+val loss/acc per WD
    wd_f2_lmc_overfit.{pdf,png}   — LMC × epoch with val_loss-min markers
    wd_f3_sampen_overfit.{pdf,png}
    wd_f4_by_group.{pdf,png}
    wd_f5_inflection_table.csv/tex
    wd_f6_lead_time.{pdf,png}     — lead-time bar chart with bootstrap 95% CI

Usage:
    python scripts/make_wd_figs.py
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
OUT         = THESIS_ROOT / "results" / "wd_figs"
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 123, 2024]
# Four WD values; the WD=1e-4 baseline reuses the existing 27-run grid runs.
WD_VALUES = [0.0, 1e-5, 1e-4, 1e-3]
EPOCHS = 100

WD_COLOR = {
    0.0:   "#d62728",   # red — no regularization
    1e-5:  "#ff7f0e",   # orange — weak
    1e-4:  "#2ca02c",   # green — baseline
    1e-3:  "#1f77b4",   # blue — strong
}
WD_LABEL = {
    0.0:   "WD=0",
    1e-5:  "WD=1e-5",
    1e-4:  "WD=1e-4 (baseline)",
    1e-3:  "WD=1e-3",
}

# Robust detector constants — same convention as Phase 1
ROBUST_GROUP    = "bn_gamma"
ROBUST_MIN_EP   = 0.5
ROBUST_WINDOW   = 7
ROBUST_POLY     = 3
ROBUST_NBOOT    = 1000
ROBUST_CI       = 0.95

plt.rcParams.update({
    "font.size": 10.5,
    "axes.titlesize": 11.5,
    "axes.labelsize": 10.5,
    "legend.fontsize": 9,
    "figure.dpi": 130,
})


def run_id(wd: float, seed: int) -> str:
    """Mirror the orchestrator's make_run_id() for the WD axis."""
    base = f"PathMNIST_{EPOCHS}ep_seed{seed}"
    if abs(wd - 1e-4) > 1e-12:
        if wd == 0.0:
            base += "_wd0"
        else:
            import math
            exp = int(round(math.log10(wd))) if wd > 0 else 0
            base += f"_wd1e{exp:+03d}"
    return base


def load_history(wd: float, seed: int) -> pd.DataFrame | None:
    p = RUNS_DIR / run_id(wd, seed) / "history.csv"
    return pd.read_csv(p) if p.exists() else None


def load_metrics(wd: float, seed: int) -> pd.DataFrame | None:
    p = RUNS_DIR / run_id(wd, seed) / "metrics.csv"
    return pd.read_csv(p) if p.exists() else None


def discover_completed():
    found = []
    for wd in WD_VALUES:
        for seed in SEEDS:
            if load_history(wd, seed) is not None and load_metrics(wd, seed) is not None:
                found.append((wd, seed))
    return found


def min_val_loss_epoch(history: pd.DataFrame) -> float:
    val = history[history["phase"] == "val"]
    if len(val) == 0:
        return float("nan")
    return float(val.loc[val["loss"].idxmin(), "epoch"])


def aggregate_metric_over_seeds(wd, metric_col, group=None):
    dfs = []
    for seed in SEEDS:
        df = load_metrics(wd, seed)
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


def _bn_gamma_late_dfs(wd):
    dfs = []
    for seed in SEEDS:
        m = load_metrics(wd, seed)
        if m is None:
            continue
        sub = m[(m["group"] == ROBUST_GROUP) & (m["epoch"] >= ROBUST_MIN_EP)]
        sub = sub[["step", "epoch", "lmc_complexity", "sampen_1d"]].copy()
        if not sub.empty:
            dfs.append(sub)
    return dfs


# ============================================================================
# F1 — Training curves by WD
# ============================================================================
def fig_val_curves():
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharex=True)
    for col, wd in enumerate(WD_VALUES):
        for seed in SEEDS:
            h = load_history(wd, seed)
            if h is None:
                continue
            tr = h[h["phase"] == "train"]
            va = h[h["phase"] == "val"]
            axes[0, col].plot(tr["epoch"], tr["loss"], alpha=0.5, linewidth=1,
                              color=WD_COLOR[wd])
            axes[0, col].plot(va["epoch"], va["loss"], "--", alpha=0.85, linewidth=1.5,
                              color=WD_COLOR[wd], label=f"seed={seed}")
            axes[1, col].plot(tr["epoch"], tr["acc"], alpha=0.5, linewidth=1,
                              color=WD_COLOR[wd])
            axes[1, col].plot(va["epoch"], va["acc"], "--", alpha=0.85, linewidth=1.5,
                              color=WD_COLOR[wd])
            e_star = min_val_loss_epoch(h)
            axes[0, col].axvline(e_star, color="black", ls=":", lw=0.8, alpha=0.4)
            axes[1, col].axvline(e_star, color="black", ls=":", lw=0.8, alpha=0.4)
        axes[0, col].set_title(WD_LABEL[wd])
        axes[0, col].set_ylabel("Loss" if col == 0 else "")
        axes[1, col].set_ylabel("Accuracy" if col == 0 else "")
        axes[1, col].set_xlabel("Época")
        axes[0, col].grid(alpha=0.3); axes[1, col].grid(alpha=0.3)
        axes[0, col].legend(fontsize=7, loc="upper right")
    fig.suptitle("Curvas de treino sob variação de weight decay — PathMNIST 100 ep (linha cheia: treino; tracejada: val)",
                 y=1.01, fontsize=12)
    plt.tight_layout()
    plt.savefig(OUT / "wd_f1_val_curves.pdf"); plt.savefig(OUT / "wd_f1_val_curves.png")
    plt.close()
    print("WD-F1 ok")


def _fig_metric_by_wd(metric_col, ylabel, title, fname):
    fig, ax = plt.subplots(figsize=(10, 5))
    for wd in WD_VALUES:
        agg = aggregate_metric_over_seeds(wd, metric_col)
        if agg is None or len(agg) == 0:
            continue
        ax.plot(agg["epoch"], agg["mean"], color=WD_COLOR[wd],
                linewidth=2, label=WD_LABEL[wd])
        ax.fill_between(agg["epoch"], agg["mean"] - agg["std"], agg["mean"] + agg["std"],
                        color=WD_COLOR[wd], alpha=0.16)
        e_stars = []
        for seed in SEEDS:
            h = load_history(wd, seed)
            if h is not None:
                e_stars.append(min_val_loss_epoch(h))
        if e_stars:
            ax.axvline(float(np.nanmean(e_stars)), color=WD_COLOR[wd], ls=":", lw=1.0, alpha=0.7)
    ax.set_xscale("symlog", linthresh=1e-2)
    ax.set_xlabel("Época (simlog)"); ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3); ax.legend()
    plt.tight_layout()
    plt.savefig(OUT / f"{fname}.pdf"); plt.savefig(OUT / f"{fname}.png")
    plt.close()


def fig_lmc():
    _fig_metric_by_wd("lmc_complexity", "LMC complexity (média entre camadas)",
                      "LMC × época por weight decay — marcador vertical: $\\arg\\min_e$ val_loss",
                      "wd_f2_lmc_overfit")
    print("WD-F2 ok")


def fig_sampen():
    _fig_metric_by_wd("sampen_1d", "SampEn (média entre camadas)",
                      "SampEn × época por weight decay",
                      "wd_f3_sampen_overfit")
    print("WD-F3 ok")


def fig_by_group():
    groups = ["conv", "dense", "bn_gamma", "bn_beta"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for ax, group in zip(axes.flat, groups):
        for wd in WD_VALUES:
            agg = aggregate_metric_over_seeds(wd, "lmc_complexity", group=group)
            if agg is None or len(agg) == 0:
                continue
            ax.plot(agg["epoch"], agg["mean"], color=WD_COLOR[wd],
                    linewidth=2, label=WD_LABEL[wd])
            ax.fill_between(agg["epoch"], agg["mean"] - agg["std"], agg["mean"] + agg["std"],
                            color=WD_COLOR[wd], alpha=0.14)
        ax.set_xscale("symlog", linthresh=1e-2)
        ax.set_title(f"Grupo {group}")
        ax.set_ylabel("LMC"); ax.set_xlabel("Época (simlog)")
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle("LMC por grupo de camadas e por weight decay — PathMNIST 100 épocas", y=1.01, fontsize=12)
    plt.tight_layout()
    plt.savefig(OUT / "wd_f4_by_group.pdf"); plt.savefig(OUT / "wd_f4_by_group.png")
    plt.close()
    print("WD-F4 ok")


def fig_summary_table():
    rows = []
    for wd in WD_VALUES:
        e_val_list, vmin_list, vacc_list = [], [], []
        for seed in SEEDS:
            h = load_history(wd, seed)
            if h is None:
                continue
            val = h[h["phase"] == "val"]
            e_val_list.append(float(val.loc[val["loss"].idxmin(), "epoch"]))
            vmin_list.append(float(val["loss"].min()))
            vacc_list.append(float(val.loc[val["loss"].idxmin(), "acc"]))

        dfs = _bn_gamma_late_dfs(wd)
        if not dfs:
            continue
        peak_lmc = bootstrap_peak_ci(dfs, "lmc_complexity",
                                     n_bootstrap=ROBUST_NBOOT, ci_level=ROBUST_CI,
                                     window_size=ROBUST_WINDOW, polyorder=ROBUST_POLY)
        peak_se  = bootstrap_peak_ci(dfs, "sampen_1d",
                                     n_bootstrap=ROBUST_NBOOT, ci_level=ROBUST_CI,
                                     window_size=ROBUST_WINDOW, polyorder=ROBUST_POLY)
        e_val_mean = float(np.nanmean(e_val_list)) if e_val_list else np.nan
        lead = e_val_mean - peak_lmc.epoch_peak if np.isfinite(peak_lmc.epoch_peak) else np.nan
        rows.append({
            "wd_label":           WD_LABEL[wd],
            "wd":                 wd,
            "epoch_min_val_loss": e_val_mean,
            "min_val_loss":       float(np.nanmean(vmin_list)) if vmin_list else np.nan,
            "val_acc_at_min":     float(np.nanmean(vacc_list)) if vacc_list else np.nan,
            "epoch_peak_lmc":     peak_lmc.epoch_peak,
            "lmc_ci_low":         peak_lmc.ci_low,
            "lmc_ci_high":        peak_lmc.ci_high,
            "epoch_peak_sampen":  peak_se.epoch_peak,
            "lead_time_lmc":      lead,
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "wd_f5_inflection_table.csv", index=False)

    lines = [
        "% Auto-generated by scripts/make_wd_figs.py",
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{Pontos de inflexão e \\emph{lead time} sob variação de weight decay (PathMNIST 100 épocas, média entre 3 sementes; \\textit{bn\\_gamma}; bootstrap 95\\% CI).}",
        "\\label{tab:wd-inflexao}",
        "\\small",
        "\\begin{tabular}{lrcrr}",
        "\\toprule",
        "Weight decay & $e^*_\\mathrm{val}$ & min val\\_loss & $e^{\\text{peak}}_\\mathrm{LMC}$ [95\\% CI] & lead time \\\\",
        "\\midrule",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"{r['wd_label']} & "
            f"{r['epoch_min_val_loss']:.1f} & "
            f"{r['min_val_loss']:.3f} & "
            f"{r['epoch_peak_lmc']:.1f} [{r['lmc_ci_low']:.1f}, {r['lmc_ci_high']:.1f}] & "
            f"{r['lead_time_lmc']:+.1f} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (OUT / "wd_f5_inflection_table.tex").write_text("\n".join(lines) + "\n")
    print("WD-F5 ok")
    print(df.to_string(index=False))


def fig_lead_time():
    means, lo, hi, labels, colors = [], [], [], [], []
    for wd in WD_VALUES:
        e_val_list = []
        for seed in SEEDS:
            h = load_history(wd, seed)
            if h is None:
                continue
            val = h[h["phase"] == "val"]
            e_val_list.append(float(val.loc[val["loss"].idxmin(), "epoch"]))
        if not e_val_list:
            continue
        e_val_mean = float(np.nanmean(e_val_list))

        dfs = _bn_gamma_late_dfs(wd)
        if not dfs:
            continue
        peak = bootstrap_peak_ci(dfs, "lmc_complexity",
                                 n_bootstrap=ROBUST_NBOOT, ci_level=ROBUST_CI,
                                 window_size=ROBUST_WINDOW, polyorder=ROBUST_POLY)
        labels.append(WD_LABEL[wd])
        colors.append(WD_COLOR[wd])
        means.append(e_val_mean - peak.epoch_peak)
        lo.append(e_val_mean - peak.ci_high)
        hi.append(e_val_mean - peak.ci_low)

    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    x = np.arange(len(labels))
    err = np.vstack([np.array(means) - np.array(lo),
                     np.array(hi) - np.array(means)])
    ax.bar(x, means, color=colors, alpha=0.78, edgecolor="black", linewidth=0.8)
    ax.errorbar(x, means, yerr=err, fmt="none", ecolor="black", capsize=6, linewidth=1.4)
    ax.axhline(0, color="gray", linewidth=0.7)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_xlabel("Weight decay")
    ax.set_ylabel(r"Lead time = $e^*_\mathrm{val} - e^{\mathrm{peak}}_\mathrm{LMC}$ (épocas)")
    ax.set_title("Lead time da LMC por weight decay (bn$_\\gamma$, epoch $\\geq 0.5$) — 95% CI bootstrap")
    ax.grid(alpha=0.3, axis="y")
    for i, (m, h) in enumerate(zip(means, hi)):
        ax.text(i, max(h, m) + 0.5, f"{m:+.1f}",
                ha="center", va="bottom", fontsize=10, weight="bold")
    plt.tight_layout()
    plt.savefig(OUT / "wd_f6_lead_time.pdf"); plt.savefig(OUT / "wd_f6_lead_time.png")
    plt.close()
    print("WD-F6 ok")


def main():
    done = discover_completed()
    print(f"Found {len(done)}/{len(WD_VALUES) * len(SEEDS)} completed runs.")
    if not done:
        print("Nothing to plot — aborting.")
        sys.exit(0)
    fig_val_curves()
    fig_lmc()
    fig_sampen()
    fig_by_group()
    fig_summary_table()
    fig_lead_time()
    print(f"\nAll figures written to: {OUT}")


if __name__ == "__main__":
    main()

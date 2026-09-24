"""Phase-1 figures: train-set subsampling × LMC/SampEn trajectories.

Reads the 9 runs of config/experiment_grid_overfit.yaml and produces:
    f1_val_curves.{pdf,png}   — train+val loss/acc per fraction
    f2_lmc_overfit.{pdf,png}  — LMC × epoch, marker at val_loss minimum
    f3_sampen_overfit.{pdf,png}
    f4_by_group.{pdf,png}     — per-layer-group breakdown (one fraction)
    f5_inflection_table.csv   — quantitative summary table
    f5_inflection_table.tex   — same as LaTeX booktabs

Usage:
    python scripts/make_overfit_figs.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Local helper
sys.path.insert(0, str(Path(__file__).resolve().parent))
from robust_detector import bootstrap_peak_ci, smoothed_mean_curve  # noqa: E402

# Robust-detector convention (justified in tese §4.4.6):
#   - canonical group: bn_gamma (cleanest signal per Phase-1 Fig. 8)
#   - restrict to epoch >= 0.5 (skip warm-up where init noise dominates)
#   - Savitzky-Golay window 7, polyorder 3
#   - bootstrap 1000 resamples for 95% CI on peak epoch
ROBUST_GROUP    = "bn_gamma"
ROBUST_MIN_EP   = 0.5
ROBUST_WINDOW   = 7
ROBUST_POLY     = 3
ROBUST_NBOOT    = 1000
ROBUST_CI       = 0.95

THESIS_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = THESIS_ROOT / "results" / "runs"
OUT = THESIS_ROOT / "results" / "overfit_figs"
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 123, 2024]
FRACTIONS = [0.05, 0.10, 0.25]
EPOCHS = 100

FRAC_COLOR = {0.05: "#d62728", 0.10: "#ff7f0e", 0.25: "#1f77b4"}
FRAC_LABEL = {0.05: "5%", 0.10: "10%", 0.25: "25%"}

plt.rcParams.update({
    "font.size": 10.5,
    "axes.titlesize": 11.5,
    "axes.labelsize": 10.5,
    "legend.fontsize": 9,
    "figure.dpi": 130,
})


def run_id(frac: float, seed: int) -> str:
    return f"PathMNIST_{EPOCHS}ep_seed{seed}_frac{frac:.2f}"


def load_history(frac: float, seed: int) -> pd.DataFrame | None:
    p = RUNS_DIR / run_id(frac, seed) / "history.csv"
    return pd.read_csv(p) if p.exists() else None


def load_metrics(frac: float, seed: int) -> pd.DataFrame | None:
    p = RUNS_DIR / run_id(frac, seed) / "metrics.csv"
    return pd.read_csv(p) if p.exists() else None


def discover_completed():
    found = []
    for frac in FRACTIONS:
        for seed in SEEDS:
            h = load_history(frac, seed)
            m = load_metrics(frac, seed)
            if h is not None and m is not None and len(h) > 0:
                found.append((frac, seed))
    return found


# ============================================================================
# Helpers
# ============================================================================
def min_val_loss_epoch(history: pd.DataFrame) -> float:
    """Return the epoch number where validation loss is minimised."""
    val = history[history["phase"] == "val"]
    if len(val) == 0:
        return float("nan")
    idx = val["loss"].idxmin()
    return float(val.loc[idx, "epoch"])


def aggregate_metric_over_seeds(frac, metric_col, group=None):
    """Return DataFrame with columns: step, epoch, mean, std (averaged over seeds)."""
    dfs = []
    for seed in SEEDS:
        df = load_metrics(frac, seed)
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


# ============================================================================
# F1 — Training curves by fraction
# ============================================================================
def fig_val_curves():
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    for col, frac in enumerate(FRACTIONS):
        for seed in SEEDS:
            h = load_history(frac, seed)
            if h is None:
                continue
            tr = h[h["phase"] == "train"]
            va = h[h["phase"] == "val"]
            axes[0, col].plot(tr["epoch"], tr["loss"], alpha=0.5, linewidth=1,
                              color=FRAC_COLOR[frac])
            axes[0, col].plot(va["epoch"], va["loss"], "--", alpha=0.85, linewidth=1.5,
                              color=FRAC_COLOR[frac], label=f"seed={seed}")
            axes[1, col].plot(tr["epoch"], tr["acc"], alpha=0.5, linewidth=1,
                              color=FRAC_COLOR[frac])
            axes[1, col].plot(va["epoch"], va["acc"], "--", alpha=0.85, linewidth=1.5,
                              color=FRAC_COLOR[frac])
            # mark val_loss minimum
            e_star = min_val_loss_epoch(h)
            axes[0, col].axvline(e_star, color="black", ls=":", lw=0.8, alpha=0.4)
            axes[1, col].axvline(e_star, color="black", ls=":", lw=0.8, alpha=0.4)
        axes[0, col].set_title(f"Train = {FRAC_LABEL[frac]} of PathMNIST")
        axes[0, col].set_ylabel("Loss" if col == 0 else "")
        axes[1, col].set_ylabel("Accuracy" if col == 0 else "")
        axes[1, col].set_xlabel("Época")
        axes[0, col].grid(alpha=0.3); axes[1, col].grid(alpha=0.3)
        axes[0, col].legend(fontsize=7, loc="upper right")
    fig.suptitle("Curvas de treino sob subamostragem (linha cheia: train; tracejada: val; "
                 "linha pontilhada vertical: mínimo da val_loss)",
                 y=1.01, fontsize=12)
    plt.tight_layout()
    plt.savefig(OUT / "f1_val_curves.pdf"); plt.savefig(OUT / "f1_val_curves.png")
    plt.close()
    print("F1 ok")


# ============================================================================
# F2 — LMC × epoch with val_loss minimum overlay
# ============================================================================
def _fig_metric_by_fraction(metric_col, ylabel, title, fname):
    fig, ax = plt.subplots(figsize=(10, 5))
    for frac in FRACTIONS:
        agg = aggregate_metric_over_seeds(frac, metric_col)
        if agg is None or len(agg) == 0:
            continue
        ax.plot(agg["epoch"], agg["mean"], color=FRAC_COLOR[frac],
                linewidth=2, label=f"train = {FRAC_LABEL[frac]}")
        ax.fill_between(agg["epoch"], agg["mean"] - agg["std"], agg["mean"] + agg["std"],
                        color=FRAC_COLOR[frac], alpha=0.18)
        # average val_loss minimum across seeds for this fraction
        e_stars = []
        for seed in SEEDS:
            h = load_history(frac, seed)
            if h is not None:
                e_stars.append(min_val_loss_epoch(h))
        if e_stars:
            e_mean = float(np.nanmean(e_stars))
            ax.axvline(e_mean, color=FRAC_COLOR[frac], ls=":", lw=1.2, alpha=0.75)
            ax.text(e_mean, ax.get_ylim()[1] * 0.98 if False else 0,
                    f" min val_loss ({FRAC_LABEL[frac]})",
                    color=FRAC_COLOR[frac], fontsize=8, rotation=90,
                    va="bottom", ha="left", alpha=0.85,
                    transform=ax.get_xaxis_transform())
    ax.set_xscale("symlog", linthresh=1e-2)
    ax.set_xlabel("Época (simlog)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3); ax.legend()
    plt.tight_layout()
    plt.savefig(OUT / f"{fname}.pdf"); plt.savefig(OUT / f"{fname}.png")
    plt.close()


def fig_lmc():
    _fig_metric_by_fraction(
        "lmc_complexity", "LMC complexity (média entre camadas)",
        "LMC × época por fração do treino — marcador vertical: $\\arg\\min_e$ val_loss",
        "f2_lmc_overfit",
    )
    print("F2 ok")


def fig_sampen():
    _fig_metric_by_fraction(
        "sampen_1d", "SampEn (média entre camadas)",
        "SampEn × época por fração do treino — marcador vertical: $\\arg\\min_e$ val_loss",
        "f3_sampen_overfit",
    )
    print("F3 ok")


# ============================================================================
# F4 — Per-group breakdown
# ============================================================================
def fig_by_group():
    groups = ["conv", "dense", "bn_gamma", "bn_beta"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for ax, group in zip(axes.flat, groups):
        for frac in FRACTIONS:
            agg = aggregate_metric_over_seeds(frac, "lmc_complexity", group=group)
            if agg is None or len(agg) == 0:
                continue
            ax.plot(agg["epoch"], agg["mean"], color=FRAC_COLOR[frac],
                    linewidth=2, label=f"train={FRAC_LABEL[frac]}")
            ax.fill_between(agg["epoch"], agg["mean"] - agg["std"], agg["mean"] + agg["std"],
                            color=FRAC_COLOR[frac], alpha=0.15)
            # val_loss minimum marker
            e_stars = [min_val_loss_epoch(load_history(frac, s))
                       for s in SEEDS if load_history(frac, s) is not None]
            if e_stars:
                ax.axvline(float(np.nanmean(e_stars)),
                           color=FRAC_COLOR[frac], ls=":", lw=1.0, alpha=0.7)
        ax.set_xscale("symlog", linthresh=1e-2)
        ax.set_title(f"Grupo {group}")
        ax.set_ylabel("LMC"); ax.set_xlabel("Época (simlog)")
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle("LMC por grupo de camadas e por fração do treino — PathMNIST 100 épocas",
                 y=1.01, fontsize=12)
    plt.tight_layout()
    plt.savefig(OUT / "f4_by_group.pdf"); plt.savefig(OUT / "f4_by_group.png")
    plt.close()
    print("F4 ok")


# ============================================================================
# F5 — Robust inflection summary table (Savitzky-Golay + bootstrap CI)
# ============================================================================
def _bn_gamma_late_dfs(frac):
    """Return per-seed dataframes restricted to bn_gamma group and epoch ≥ MIN_EP."""
    dfs = []
    for seed in SEEDS:
        m = load_metrics(frac, seed)
        if m is None:
            continue
        sub = m[(m["group"] == ROBUST_GROUP) & (m["epoch"] >= ROBUST_MIN_EP)]
        sub = sub[["step", "epoch", "lmc_complexity", "sampen_1d"]].copy()
        if not sub.empty:
            dfs.append(sub)
    return dfs


def fig_summary_table():
    rows = []
    for frac in FRACTIONS:
        # val loss min across seeds
        e_val_list, vmin_list, vacc_list = [], [], []
        for seed in SEEDS:
            h = load_history(frac, seed)
            if h is None:
                continue
            val = h[h["phase"] == "val"]
            e_val = float(val.loc[val["loss"].idxmin(), "epoch"])
            vmin = float(val["loss"].min())
            vacc = float(val.loc[val["loss"].idxmin(), "acc"])
            e_val_list.append(e_val); vmin_list.append(vmin); vacc_list.append(vacc)

        # Robust peak detection on bn_gamma, epoch ≥ MIN_EP
        dfs = _bn_gamma_late_dfs(frac)
        peak_lmc = bootstrap_peak_ci(dfs, "lmc_complexity",
                                     n_bootstrap=ROBUST_NBOOT, ci_level=ROBUST_CI,
                                     window_size=ROBUST_WINDOW, polyorder=ROBUST_POLY)
        peak_se = bootstrap_peak_ci(dfs, "sampen_1d",
                                    n_bootstrap=ROBUST_NBOOT, ci_level=ROBUST_CI,
                                    window_size=ROBUST_WINDOW, polyorder=ROBUST_POLY)

        e_val_mean = float(np.nanmean(e_val_list)) if e_val_list else np.nan
        lead_time = e_val_mean - peak_lmc.epoch_peak if np.isfinite(peak_lmc.epoch_peak) else np.nan

        rows.append({
            "fraction":              FRAC_LABEL[frac],
            "epoch_min_val_loss":    e_val_mean,
            "min_val_loss":          float(np.nanmean(vmin_list)) if vmin_list else np.nan,
            "val_acc_at_min":        float(np.nanmean(vacc_list)) if vacc_list else np.nan,
            "epoch_peak_lmc":        peak_lmc.epoch_peak,
            "lmc_ci_low":            peak_lmc.ci_low,
            "lmc_ci_high":           peak_lmc.ci_high,
            "epoch_peak_sampen":     peak_se.epoch_peak,
            "sampen_ci_low":         peak_se.ci_low,
            "sampen_ci_high":        peak_se.ci_high,
            "lead_time_lmc":         lead_time,
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "f5_inflection_table.csv", index=False)

    # LaTeX (booktabs) variant with bootstrap CIs
    lines = [
        "% Auto-generated by scripts/make_overfit_figs.py",
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{Pontos de inflexão e \\emph{lead time} sob subamostragem do treino (PathMNIST 100 épocas, média entre 3 sementes; \\textit{bn\\_gamma}; bootstrap 95\\% CI).}",
        "\\label{tab:overfit-inflexao}",
        "\\small",
        "\\begin{tabular}{lrcrcr}",
        "\\toprule",
        "Fração & $e^*_\\mathrm{val}$ & min val\\_loss & $e^{\\text{peak}}_\\mathrm{LMC}$ [95\\% CI] & $e^{\\text{peak}}_\\mathrm{SampEn}$ [95\\% CI] & lead time \\\\",
        "\\midrule",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"{r['fraction'].replace('%', '\\%')} & "
            f"{r['epoch_min_val_loss']:.1f} & "
            f"{r['min_val_loss']:.3f} & "
            f"{r['epoch_peak_lmc']:.1f} [{r['lmc_ci_low']:.1f}, {r['lmc_ci_high']:.1f}] & "
            f"{r['epoch_peak_sampen']:.1f} [{r['sampen_ci_low']:.1f}, {r['sampen_ci_high']:.1f}] & "
            f"{r['lead_time_lmc']:+.1f} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (OUT / "f5_inflection_table.tex").write_text("\n".join(lines) + "\n")
    print("F5 ok")
    print(df.to_string(index=False))


# ============================================================================
# F6 — Lead-time chart with bootstrap CI
# ============================================================================
def fig_lead_time():
    """Bar chart of LMC lead time (e*_val - e_peak^LMC) with 95% CI per fraction.

    Uses the robust detector on bn_gamma (epoch >= ROBUST_MIN_EP).
    """
    fracs_done = []
    lead_means = []
    lead_lo = []
    lead_hi = []
    e_peak_means = []

    for frac in FRACTIONS:
        # e*_val (mean across seeds)
        e_val_list = []
        for seed in SEEDS:
            h = load_history(frac, seed)
            if h is None:
                continue
            val = h[h["phase"] == "val"]
            e_val_list.append(float(val.loc[val["loss"].idxmin(), "epoch"]))
        if not e_val_list:
            continue
        e_val_mean = float(np.nanmean(e_val_list))

        dfs = _bn_gamma_late_dfs(frac)
        if not dfs:
            continue
        peak = bootstrap_peak_ci(dfs, "lmc_complexity",
                                 n_bootstrap=ROBUST_NBOOT, ci_level=ROBUST_CI,
                                 window_size=ROBUST_WINDOW, polyorder=ROBUST_POLY)

        fracs_done.append(FRAC_LABEL[frac])
        e_peak_means.append(peak.epoch_peak)
        lead_means.append(e_val_mean - peak.epoch_peak)
        # CI on lead time = e_val_mean - CI on peak (flipped)
        lead_lo.append(e_val_mean - peak.ci_high)
        lead_hi.append(e_val_mean - peak.ci_low)

    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    x = np.arange(len(fracs_done))
    err = np.vstack([np.array(lead_means) - np.array(lead_lo),
                     np.array(lead_hi) - np.array(lead_means)])
    bars = ax.bar(x, lead_means, color=[FRAC_COLOR[f] for f in FRACTIONS[:len(fracs_done)]],
                  alpha=0.75, edgecolor="black", linewidth=0.8)
    ax.errorbar(x, lead_means, yerr=err, fmt="none", ecolor="black",
                capsize=6, linewidth=1.4)
    ax.axhline(0, color="gray", linewidth=0.7)
    ax.set_xticks(x); ax.set_xticklabels(fracs_done)
    ax.set_xlabel("Fração do conjunto de treino")
    ax.set_ylabel(r"Lead time = $e^*_\mathrm{val} - e^{\mathrm{peak}}_\mathrm{LMC}$ (épocas)")
    ax.set_title("Lead time da LMC (bn$_\\gamma$, epoch $\\geq 0.5$) — 95% CI bootstrap")
    ax.grid(alpha=0.3, axis="y")
    for i, (m, lo, hi) in enumerate(zip(lead_means, lead_lo, lead_hi)):
        ax.text(i, max(hi, m) + 0.4, f"{m:+.1f}",
                ha="center", va="bottom", fontsize=10, weight="bold")
    plt.tight_layout()
    plt.savefig(OUT / "f6_lead_time.pdf"); plt.savefig(OUT / "f6_lead_time.png")
    plt.close()
    print("F6 ok")


# ============================================================================
def main():
    done = discover_completed()
    print(f"Found {len(done)}/9 completed runs (need at least 1 per fraction).")
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

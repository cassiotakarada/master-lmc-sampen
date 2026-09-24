"""Phase-3 figures: architecture sweep × LMC/SampEn trajectories.

Reads the 6 runs of config/experiment_grid_arch.yaml plus the 3 baseline
runs (ResNet-18 with WD=1e-4, full data, 100 ep) from the original grid, and
produces architecture-comparison figures using the same robust detector
methodology established in Phase 1.

Outputs (in results/arch_figs/):
    arch_f1_val_curves.{pdf,png}     — train+val loss/acc per architecture
    arch_f2_lmc_overfit.{pdf,png}    — LMC × epoch by architecture
    arch_f3_sampen_overfit.{pdf,png}
    arch_f4_by_group.{pdf,png}       — per-layer-group breakdown
    arch_f5_inflection_table.csv/tex
    arch_f6_lead_time.{pdf,png}      — lead time per architecture with CI bars
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
OUT         = THESIS_ROOT / "results" / "arch_figs"
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 123, 2024]
ARCHS = ["resnet18", "resnet34", "densenet121"]
EPOCHS = 100

ARCH_COLOR = {
    "resnet18":    "#2ca02c",   # green — baseline
    "resnet34":    "#9467bd",   # purple
    "densenet121": "#17becf",   # teal
}
ARCH_LABEL = {
    "resnet18":    "ResNet-18 (baseline, ~11M)",
    "resnet34":    "ResNet-34 (~21M)",
    "densenet121": "DenseNet-121 (~7M)",
}

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


def run_id(arch: str, seed: int) -> str:
    """Mirror orchestrator.make_run_id() for the architecture axis."""
    base = f"PathMNIST_{EPOCHS}ep_seed{seed}"
    if arch != "resnet18":
        base += f"_arch-{arch}"
    return base


def load_history(arch, seed):
    p = RUNS_DIR / run_id(arch, seed) / "history.csv"
    return pd.read_csv(p) if p.exists() else None


def load_metrics(arch, seed):
    p = RUNS_DIR / run_id(arch, seed) / "metrics.csv"
    return pd.read_csv(p) if p.exists() else None


def discover_completed():
    found = []
    for arch in ARCHS:
        for seed in SEEDS:
            if load_history(arch, seed) is not None and load_metrics(arch, seed) is not None:
                found.append((arch, seed))
    return found


def min_val_loss_epoch(history):
    val = history[history["phase"] == "val"]
    if len(val) == 0:
        return float("nan")
    return float(val.loc[val["loss"].idxmin(), "epoch"])


def aggregate_metric_over_seeds(arch, metric_col, group=None):
    dfs = []
    for seed in SEEDS:
        df = load_metrics(arch, seed)
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


def _bn_gamma_late_dfs(arch):
    dfs = []
    for seed in SEEDS:
        m = load_metrics(arch, seed)
        if m is None:
            continue
        sub = m[(m["group"] == ROBUST_GROUP) & (m["epoch"] >= ROBUST_MIN_EP)]
        sub = sub[["step", "epoch", "lmc_complexity", "sampen_1d"]].copy()
        if not sub.empty:
            dfs.append(sub)
    return dfs


# ----------------------------------------------------------------------
def fig_val_curves():
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    for col, arch in enumerate(ARCHS):
        for seed in SEEDS:
            h = load_history(arch, seed)
            if h is None:
                continue
            tr = h[h["phase"] == "train"]; va = h[h["phase"] == "val"]
            axes[0, col].plot(tr["epoch"], tr["loss"], alpha=0.5, linewidth=1, color=ARCH_COLOR[arch])
            axes[0, col].plot(va["epoch"], va["loss"], "--", alpha=0.85, linewidth=1.5,
                              color=ARCH_COLOR[arch], label=f"seed={seed}")
            axes[1, col].plot(tr["epoch"], tr["acc"], alpha=0.5, linewidth=1, color=ARCH_COLOR[arch])
            axes[1, col].plot(va["epoch"], va["acc"], "--", alpha=0.85, linewidth=1.5, color=ARCH_COLOR[arch])
            e_star = min_val_loss_epoch(h)
            axes[0, col].axvline(e_star, color="black", ls=":", lw=0.8, alpha=0.4)
            axes[1, col].axvline(e_star, color="black", ls=":", lw=0.8, alpha=0.4)
        axes[0, col].set_title(ARCH_LABEL[arch])
        axes[0, col].set_ylabel("Loss" if col == 0 else "")
        axes[1, col].set_ylabel("Accuracy" if col == 0 else "")
        axes[1, col].set_xlabel("Época")
        axes[0, col].grid(alpha=0.3); axes[1, col].grid(alpha=0.3)
        axes[0, col].legend(fontsize=7, loc="upper right")
    fig.suptitle("Curvas de treino por arquitetura — PathMNIST 100 ep", y=1.01, fontsize=12)
    plt.tight_layout()
    plt.savefig(OUT / "arch_f1_val_curves.pdf"); plt.savefig(OUT / "arch_f1_val_curves.png")
    plt.close()
    print("ARCH-F1 ok")


def _fig_metric_by_arch(metric_col, ylabel, title, fname):
    fig, ax = plt.subplots(figsize=(10, 5))
    for arch in ARCHS:
        agg = aggregate_metric_over_seeds(arch, metric_col)
        if agg is None or len(agg) == 0:
            continue
        ax.plot(agg["epoch"], agg["mean"], color=ARCH_COLOR[arch], linewidth=2, label=ARCH_LABEL[arch])
        ax.fill_between(agg["epoch"], agg["mean"] - agg["std"], agg["mean"] + agg["std"],
                        color=ARCH_COLOR[arch], alpha=0.16)
        e_stars = [min_val_loss_epoch(load_history(arch, s)) for s in SEEDS if load_history(arch, s) is not None]
        if e_stars:
            ax.axvline(float(np.nanmean(e_stars)), color=ARCH_COLOR[arch], ls=":", lw=1.0, alpha=0.7)
    ax.set_xscale("symlog", linthresh=1e-2)
    ax.set_xlabel("Época (simlog)"); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.grid(alpha=0.3); ax.legend()
    plt.tight_layout()
    plt.savefig(OUT / f"{fname}.pdf"); plt.savefig(OUT / f"{fname}.png")
    plt.close()


def fig_lmc():
    _fig_metric_by_arch("lmc_complexity", "LMC complexity (média entre camadas)",
                        "LMC × época por arquitetura — marcador vertical: $\\arg\\min_e$ val_loss",
                        "arch_f2_lmc_overfit")
    print("ARCH-F2 ok")


def fig_sampen():
    _fig_metric_by_arch("sampen_1d", "SampEn (média entre camadas)",
                        "SampEn × época por arquitetura",
                        "arch_f3_sampen_overfit")
    print("ARCH-F3 ok")


def fig_by_group():
    groups = ["conv", "dense", "bn_gamma", "bn_beta"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for ax, group in zip(axes.flat, groups):
        for arch in ARCHS:
            agg = aggregate_metric_over_seeds(arch, "lmc_complexity", group=group)
            if agg is None or len(agg) == 0:
                continue
            ax.plot(agg["epoch"], agg["mean"], color=ARCH_COLOR[arch], linewidth=2, label=ARCH_LABEL[arch])
            ax.fill_between(agg["epoch"], agg["mean"] - agg["std"], agg["mean"] + agg["std"],
                            color=ARCH_COLOR[arch], alpha=0.14)
        ax.set_xscale("symlog", linthresh=1e-2)
        ax.set_title(f"Grupo {group}")
        ax.set_ylabel("LMC"); ax.set_xlabel("Época (simlog)")
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle("LMC por grupo de camadas e por arquitetura — PathMNIST 100 épocas",
                 y=1.01, fontsize=12)
    plt.tight_layout()
    plt.savefig(OUT / "arch_f4_by_group.pdf"); plt.savefig(OUT / "arch_f4_by_group.png")
    plt.close()
    print("ARCH-F4 ok")


def fig_summary_table():
    rows = []
    for arch in ARCHS:
        e_val_list, vmin_list, vacc_list = [], [], []
        for seed in SEEDS:
            h = load_history(arch, seed)
            if h is None:
                continue
            val = h[h["phase"] == "val"]
            e_val_list.append(float(val.loc[val["loss"].idxmin(), "epoch"]))
            vmin_list.append(float(val["loss"].min()))
            vacc_list.append(float(val.loc[val["loss"].idxmin(), "acc"]))
        dfs = _bn_gamma_late_dfs(arch)
        if not dfs:
            continue
        peak = bootstrap_peak_ci(dfs, "lmc_complexity",
                                 n_bootstrap=ROBUST_NBOOT, ci_level=ROBUST_CI,
                                 window_size=ROBUST_WINDOW, polyorder=ROBUST_POLY)
        e_val_mean = float(np.nanmean(e_val_list)) if e_val_list else np.nan
        lead = e_val_mean - peak.epoch_peak if np.isfinite(peak.epoch_peak) else np.nan
        rows.append({
            "arch_label":         ARCH_LABEL[arch],
            "arch":               arch,
            "epoch_min_val_loss": e_val_mean,
            "min_val_loss":       float(np.nanmean(vmin_list)) if vmin_list else np.nan,
            "val_acc_at_min":     float(np.nanmean(vacc_list)) if vacc_list else np.nan,
            "epoch_peak_lmc":     peak.epoch_peak,
            "lmc_ci_low":         peak.ci_low,
            "lmc_ci_high":        peak.ci_high,
            "lead_time_lmc":      lead,
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "arch_f5_inflection_table.csv", index=False)

    lines = [
        "% Auto-generated by scripts/make_arch_figs.py",
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{Pontos de inflexão e \\emph{lead time} por arquitetura (PathMNIST 100 épocas, média entre 3 sementes; \\textit{bn\\_gamma}; bootstrap 95\\% CI).}",
        "\\label{tab:arch-inflexao}",
        "\\small",
        "\\begin{tabular}{lrcrr}",
        "\\toprule",
        "Arquitetura & $e^*_\\mathrm{val}$ & min val\\_loss & $e^{\\text{peak}}_\\mathrm{LMC}$ [95\\% CI] & lead time \\\\",
        "\\midrule",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"{r['arch_label']} & "
            f"{r['epoch_min_val_loss']:.1f} & "
            f"{r['min_val_loss']:.3f} & "
            f"{r['epoch_peak_lmc']:.1f} [{r['lmc_ci_low']:.1f}, {r['lmc_ci_high']:.1f}] & "
            f"{r['lead_time_lmc']:+.1f} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (OUT / "arch_f5_inflection_table.tex").write_text("\n".join(lines) + "\n")
    print("ARCH-F5 ok")
    print(df.to_string(index=False))


def fig_lead_time():
    means, lo, hi, labels, colors = [], [], [], [], []
    for arch in ARCHS:
        e_val_list = []
        for seed in SEEDS:
            h = load_history(arch, seed)
            if h is None:
                continue
            val = h[h["phase"] == "val"]
            e_val_list.append(float(val.loc[val["loss"].idxmin(), "epoch"]))
        if not e_val_list:
            continue
        e_val_mean = float(np.nanmean(e_val_list))
        dfs = _bn_gamma_late_dfs(arch)
        if not dfs:
            continue
        peak = bootstrap_peak_ci(dfs, "lmc_complexity",
                                 n_bootstrap=ROBUST_NBOOT, ci_level=ROBUST_CI,
                                 window_size=ROBUST_WINDOW, polyorder=ROBUST_POLY)
        labels.append(ARCH_LABEL[arch].split(" (")[0])  # short label
        colors.append(ARCH_COLOR[arch])
        means.append(e_val_mean - peak.epoch_peak)
        lo.append(e_val_mean - peak.ci_high)
        hi.append(e_val_mean - peak.ci_low)

    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    x = np.arange(len(labels))
    err = np.vstack([np.array(means) - np.array(lo),
                     np.array(hi) - np.array(means)])
    ax.bar(x, means, color=colors, alpha=0.78, edgecolor="black", linewidth=0.8)
    ax.errorbar(x, means, yerr=err, fmt="none", ecolor="black", capsize=6, linewidth=1.4)
    ax.axhline(0, color="gray", linewidth=0.7)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_xlabel("Arquitetura")
    ax.set_ylabel(r"Lead time = $e^*_\mathrm{val} - e^{\mathrm{peak}}_\mathrm{LMC}$ (épocas)")
    ax.set_title("Lead time da LMC por arquitetura (bn$_\\gamma$, epoch $\\geq 0.5$) — 95% CI bootstrap")
    ax.grid(alpha=0.3, axis="y")
    for i, (m, h) in enumerate(zip(means, hi)):
        ax.text(i, max(h, m) + 0.5, f"{m:+.1f}",
                ha="center", va="bottom", fontsize=10, weight="bold")
    plt.tight_layout()
    plt.savefig(OUT / "arch_f6_lead_time.pdf"); plt.savefig(OUT / "arch_f6_lead_time.png")
    plt.close()
    print("ARCH-F6 ok")


def main():
    done = discover_completed()
    print(f"Found {len(done)}/{len(ARCHS) * len(SEEDS)} completed runs.")
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

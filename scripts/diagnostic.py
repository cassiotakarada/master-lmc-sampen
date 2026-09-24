#!/usr/bin/env python3
"""
scripts/diagnostic.py — ~5-minute smoke test of the full pipeline.

Trains ResNet-18 on PathMNIST for 2 epochs (10 log-spaced checkpoints),
collects LMC + SampEn₁D metrics per layer group, saves diagnostic plots,
and prints a sanity-check report.

Usage:
    python scripts/diagnostic.py [--force] [--data-root DIR]

Options:
    --force       Re-run even if the diagnostic is already complete
    --data-root   Directory for MedMNIST downloads (default: ./data)

Exit codes:
    0  all sanity checks passed
    1  training error or one or more checks failed
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Make 'src' importable when the script is executed directly
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use('Agg')   # headless — no display needed on GPU servers
import matplotlib.pyplot as plt
import pandas as pd

from src.training import train_run

log = logging.getLogger('diagnostic')

# ─── Config ───────────────────────────────────────────────────────────────────

DIAG_CONFIG: dict = {
    'run_id':          'diagnostic',
    'dataset':         'PathMNIST',
    'epochs':          2,
    'seed':            42,
    'lr':              0.001,
    'momentum':        0.9,
    'weight_decay':    0.0001,
    'batch_size':      64,
    'image_size':      64,
    'n_checkpoints':   10,
    'max_flat_sample': 2_000,   # smaller sample keeps SampEn₁D fast on any machine
    'num_workers':     4,
}

_GROUP_COLORS = {
    'conv':     '#1f77b4',
    'dense':    '#ff7f0e',
    'bn_gamma': '#2ca02c',
    'bn_beta':  '#d62728',
}


# ─── Sanity checks ────────────────────────────────────────────────────────────

def _ok(label: str, cond: bool) -> bool:
    tag = '\033[92m[PASS]\033[0m' if cond else '\033[91m[FAIL]\033[0m'
    print(f'  {tag}  {label}')
    return cond


def sanity_checks(metrics: pd.DataFrame, history: pd.DataFrame) -> bool:
    """
    Run all sanity checks against the collected metrics and training history.
    Prints each result and returns True only if every check passes.
    """
    results: list[bool] = []

    results.append(_ok('metrics.csv is non-empty', len(metrics) > 0))
    results.append(_ok('history.csv is non-empty', len(history) > 0))

    if not metrics.empty:
        results.append(_ok(
            "groups 'conv' and 'dense' present",
            {'conv', 'dense'}.issubset(set(metrics['group'].unique())),
        ))
        results.append(_ok(
            'LMC complexity > 0 for all rows',
            bool((metrics['lmc_complexity'] > 0).all()),
        ))
        results.append(_ok(
            'L2 norm mean ≥ 0 for all rows',
            bool((metrics['l2_mean'] >= 0).all()),
        ))

        se_numeric = pd.to_numeric(metrics['sampen_1d'], errors='coerce')
        finite_frac = float(se_numeric.notna().mean())
        results.append(_ok(
            f'SampEn₁D is finite for ≥ 50 % of rows  (got {finite_frac:.0%})',
            finite_frac >= 0.5,
        ))

    if not history.empty:
        val_rows = history[history['phase'] == 'val']
        if not val_rows.empty:
            final_acc = float(val_rows.sort_values('epoch').iloc[-1]['acc'])
            results.append(_ok(
                f'Final val_acc ∈ [0, 1]  (got {final_acc:.4f})',
                0.0 <= final_acc <= 1.0,
            ))

    return all(results)


# ─── Plots ────────────────────────────────────────────────────────────────────

def make_plots(
    metrics: pd.DataFrame,
    history: pd.DataFrame,
    plot_dir: Path,
) -> Path | None:
    """
    Generate a 2×2 diagnostic figure and save it as diagnostic.png.
    Returns the path to the saved file, or None if there is nothing to plot.
    """
    if metrics.empty:
        log.warning('No metrics to plot')
        return None

    plot_dir.mkdir(parents=True, exist_ok=True)
    groups = sorted(metrics['group'].unique())

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Diagnostic — PathMNIST, 2 epochs', fontsize=12, y=1.01)

    # ── LMC complexity ──────────────────────────────────────────────────────
    ax = axes[0, 0]
    for grp in groups:
        sub = metrics[metrics['group'] == grp].sort_values('step')
        ax.plot(sub['step'], sub['lmc_complexity'],
                label=grp, color=_GROUP_COLORS.get(grp),
                marker='o', markersize=4, linewidth=1.5)
    ax.set_xlabel('Training step')
    ax.set_ylabel('C = H × D  (bits × L2)')
    ax.set_title('LMC Statistical Complexity')
    ax.legend(fontsize=8)

    # ── SampEn₁D ────────────────────────────────────────────────────────────
    ax = axes[0, 1]
    for grp in groups:
        sub = metrics[metrics['group'] == grp].sort_values('step')
        se  = pd.to_numeric(sub['sampen_1d'], errors='coerce')
        mask = se.notna()
        ax.plot(sub['step'][mask], se[mask],
                label=grp, color=_GROUP_COLORS.get(grp),
                marker='o', markersize=4, linewidth=1.5)
    ax.set_xlabel('Training step')
    ax.set_ylabel('SampEn (nats)')
    ax.set_title('Sample Entropy 1D')
    ax.legend(fontsize=8)

    # ── L2 norm mean ────────────────────────────────────────────────────────
    ax = axes[1, 0]
    for grp in groups:
        sub = metrics[metrics['group'] == grp].sort_values('step')
        ax.plot(sub['step'], sub['l2_mean'],
                label=grp, color=_GROUP_COLORS.get(grp),
                marker='o', markersize=4, linewidth=1.5)
    ax.set_xlabel('Training step')
    ax.set_ylabel('Mean L2 norm per unit')
    ax.set_title('L2 Norm Trajectory')
    ax.legend(fontsize=8)

    # ── Train / val accuracy ─────────────────────────────────────────────────
    ax = axes[1, 1]
    if not history.empty:
        for phase, ls in [('train', '-'), ('val', '--')]:
            sub = history[history['phase'] == phase].sort_values('epoch')
            ax.plot(sub['epoch'], pd.to_numeric(sub['acc'], errors='coerce'),
                    linestyle=ls, marker='o', markersize=5, label=phase)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0, 1)
    ax.set_title('Train / Val Accuracy')
    ax.legend()

    plt.tight_layout()
    out = plot_dir / 'diagnostic.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(argv=None):
    p = argparse.ArgumentParser(
        description='~5-minute smoke test of the complexity-NN pipeline.'
    )
    p.add_argument('--force',     action='store_true',
                   help='Re-run even if the diagnostic is already complete')
    p.add_argument('--data-root', default='./data',
                   help='Directory for MedMNIST downloads (default: ./data)')
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s  %(levelname)s  %(message)s',
        datefmt='%H:%M:%S',
    )

    diag_dir = Path('results') / 'diagnostic'
    plot_dir = diag_dir / 'plots'
    cfg      = {**DIAG_CONFIG, 'data_root': args.data_root}

    log.info('=' * 60)
    log.info('Diagnostic smoke test — %s', cfg['dataset'])
    log.info('Epochs=%d  checkpoints=%d  max_flat=%d',
             cfg['epochs'], cfg['n_checkpoints'], cfg['max_flat_sample'])
    log.info('Output: %s', diag_dir.resolve())
    log.info('=' * 60)

    # ── Training ─────────────────────────────────────────────────────────────
    t_train_start = time.perf_counter()
    try:
        result = train_run(cfg, output_dir=diag_dir, resume=not args.force)
    except Exception as exc:
        log.error('train_run failed: %s', exc, exc_info=True)
        sys.exit(1)
    train_elapsed = time.perf_counter() - t_train_start

    log.info('Training done in %.1f s', train_elapsed)

    # ── Load outputs ──────────────────────────────────────────────────────────
    metrics_path = Path(result['metrics_path'])
    history_path = Path(result['history_path'])

    try:
        metrics = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()
        history = pd.read_csv(history_path) if history_path.exists() else pd.DataFrame()
    except Exception as exc:
        log.error('Failed to read output CSVs: %s', exc)
        sys.exit(1)

    # ── Sanity checks ────────────────────────────────────────────────────────
    print()
    print('Sanity checks:')
    all_ok = sanity_checks(metrics, history)
    print()

    # ── Plots ────────────────────────────────────────────────────────────────
    plot_path = make_plots(metrics, history, plot_dir)
    if plot_path:
        log.info('Plot saved → %s', plot_path)

    # ── Summary ───────────────────────────────────────────────────────────────
    total_elapsed = time.perf_counter() - t_train_start
    val_rows = history[history['phase'] == 'val'] if not history.empty else pd.DataFrame()
    final_acc  = float(val_rows.sort_values('epoch').iloc[-1]['acc']) \
                 if not val_rows.empty else float('nan')
    final_loss = float(val_rows.sort_values('epoch').iloc[-1]['loss']) \
                 if not val_rows.empty else float('nan')

    print('Summary:')
    print(f'  Total elapsed : {total_elapsed:.1f} s')
    print(f'  Training time : {train_elapsed:.1f} s')
    print(f'  Metric rows   : {len(metrics)}')
    print(f'  History rows  : {len(history)}')
    print(f'  Final val_acc : {final_acc:.4f}')
    print(f'  Final val_loss: {final_loss:.4f}')
    if not metrics.empty:
        print(f'  Groups        : {sorted(metrics["group"].unique())}')
        se_num = pd.to_numeric(metrics['sampen_1d'], errors='coerce')
        for grp in sorted(metrics['group'].unique()):
            mask = metrics['group'] == grp
            lmc_mean = metrics.loc[mask, 'lmc_complexity'].mean()
            se_mean  = se_num[mask].mean()
            print(f'    {grp:10s}  LMC={lmc_mean:.5f}  SE1D={se_mean:.4f}')
    if plot_path:
        print(f'  Plot          : {plot_path}')
    print()

    if not all_ok:
        log.error('One or more sanity checks FAILED — see above')
        sys.exit(1)

    log.info('All checks passed ✓')
    sys.exit(0)


if __name__ == '__main__':
    main()

"""Robust peak/inflection detection for LMC and SampEn trajectories.

The naïve ``argmax`` of a per-seed average is fragile under noise: the
5%-subsample trajectories of Phase 1 illustrate this --- ``argmax`` lands
near initialisation when the trajectory is essentially flat with noise.
This module provides:

1. Savitzky-Golay smoothing of a per-checkpoint trajectory.
2. Peak-of-smoothed estimator (point estimate).
3. Bootstrap-over-seeds for confidence intervals on the peak epoch.

The signal-processing convention (Savitzky-Golay) is consistent with the
broader thesis context (entropy of physiological time series).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


# ============================================================================
# Smoothing
# ============================================================================

def smooth_trajectory(
    values: np.ndarray,
    window_size: int = 11,
    polyorder: int = 3,
) -> np.ndarray:
    """Savitzky-Golay smoothing with automatic window adjustment.

    The window is clamped to the largest odd integer ≤ min(window_size, len(values)),
    and ``polyorder`` is clamped to ``window - 1``.  Series shorter than 5
    samples are returned unmodified.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n < 5:
        return values.copy()
    w = min(window_size, n)
    if w % 2 == 0:
        w -= 1
    p = min(polyorder, max(1, w - 1))
    return savgol_filter(values, window_length=w, polyorder=p)


# ============================================================================
# Point estimate
# ============================================================================

def find_peak_smoothed(
    epochs: np.ndarray,
    values: np.ndarray,
    window_size: int = 11,
    polyorder: int = 3,
) -> tuple[float, float]:
    """Return (epoch_at_peak, value_at_peak) of the smoothed trajectory."""
    if len(values) == 0:
        return float("nan"), float("nan")
    smoothed = smooth_trajectory(values, window_size, polyorder)
    idx = int(np.argmax(smoothed))
    return float(epochs[idx]), float(smoothed[idx])


# ============================================================================
# Bootstrap CI
# ============================================================================

@dataclass
class PeakEstimate:
    """Holds a point estimate of the peak epoch plus a bootstrap CI."""
    epoch_peak: float
    ci_low: float
    ci_high: float
    n_bootstrap: int
    ci_level: float

    def as_str(self, decimals: int = 1) -> str:
        return (
            f"{self.epoch_peak:.{decimals}f} "
            f"[{self.ci_low:.{decimals}f}, {self.ci_high:.{decimals}f}]"
        )


def _stack_seeds(per_seed_dfs: Sequence[pd.DataFrame],
                 metric_col: str) -> tuple[np.ndarray, np.ndarray]:
    """Align seeds on their common step grid; return (epochs, matrix).

    matrix has shape (n_seeds, n_steps).  epochs is the per-step epoch (mean
    across seeds, since the same step typically maps to the same epoch).
    """
    if len(per_seed_dfs) == 0:
        return np.array([]), np.zeros((0, 0))

    step_sets = [set(df["step"]) for df in per_seed_dfs]
    common = sorted(set.intersection(*step_sets))
    if not common:
        return np.array([]), np.zeros((0, 0))

    matrix = np.zeros((len(per_seed_dfs), len(common)))
    epoch_accum = np.zeros(len(common))
    step_idx = {s: i for i, s in enumerate(common)}
    for i, df in enumerate(per_seed_dfs):
        df_filt = df[df["step"].isin(common)].sort_values("step")
        matrix[i] = df_filt[metric_col].values
        epoch_accum += df_filt["epoch"].values
    epochs = epoch_accum / len(per_seed_dfs)
    return epochs, matrix


def bootstrap_peak_ci(
    per_seed_dfs: Sequence[pd.DataFrame],
    metric_col: str,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    window_size: int = 11,
    polyorder: int = 3,
    rng_seed: int = 0,
) -> PeakEstimate:
    """Bootstrap the peak epoch by resampling seeds with replacement.

    Returns ``PeakEstimate`` with the point estimate (peak of smoothed mean
    over the full sample) and a percentile interval at ``ci_level``.
    """
    epochs, matrix = _stack_seeds(per_seed_dfs, metric_col)
    if matrix.size == 0:
        return PeakEstimate(float("nan"), float("nan"), float("nan"),
                            n_bootstrap, ci_level)

    n_seeds = matrix.shape[0]
    # Point estimate: smoothed mean curve, argmax
    mean_curve = matrix.mean(axis=0)
    e_point, _ = find_peak_smoothed(epochs, mean_curve, window_size, polyorder)

    # Bootstrap over seeds (resampled with replacement)
    rng = np.random.default_rng(rng_seed)
    peaks = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n_seeds, n_seeds)
        curve = matrix[idx].mean(axis=0)
        e_peak, _ = find_peak_smoothed(epochs, curve, window_size, polyorder)
        peaks[b] = e_peak

    alpha = (1.0 - ci_level) / 2.0
    lo = float(np.percentile(peaks, 100 * alpha))
    hi = float(np.percentile(peaks, 100 * (1.0 - alpha)))
    return PeakEstimate(e_point, lo, hi, n_bootstrap, ci_level)


def smoothed_mean_curve(
    per_seed_dfs: Sequence[pd.DataFrame],
    metric_col: str,
    window_size: int = 11,
    polyorder: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (epochs, raw_mean, smoothed_mean) for plotting."""
    epochs, matrix = _stack_seeds(per_seed_dfs, metric_col)
    if matrix.size == 0:
        return np.array([]), np.array([]), np.array([])
    raw = matrix.mean(axis=0)
    sm = smooth_trajectory(raw, window_size, polyorder)
    return epochs, raw, sm

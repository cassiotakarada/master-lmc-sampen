"""
src/complexity.py — Statistical complexity measures for NN weight analysis.

Implements:
  - LMC statistical complexity    (López-Ruiz, Mancini & Calbet, 1995)
  - 1D Sample Entropy             (Richman & Moorman, 2000; Pincus, 1991)
  - 1D Multiscale Sample Entropy  (Costa, Goldberger & Peng, 2002 / 2005)

All functions operate on numpy arrays and are pure (no side effects).
Default parameters match the conventions validated by the research group:
  - SampEn: m=2, r = 0.2 × SD(series, ddof=0)   [supervisor's validated code]
  - MSE: max_scale=10 with r fixed from the original (τ=1) series
  - LMC: Shannon entropy in bits (log₂), unnormalized product H × D
         bins = min(100, max(10, n // 5))          [supervisor's validated code]

Design notes:
  - MSE uses r computed from the original series and keeps it fixed across
    scales, as in Costa et al. (2002).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist
from typing import Optional


# ============================================================================
# Private helpers
# ============================================================================

def _coarse_grain_1d(x: np.ndarray, tau: int) -> np.ndarray:
    """
    Coarse-grain a 1D series by averaging over non-overlapping tau-length blocks.
    Trailing samples that do not fill a complete block are discarded.
    """
    if tau == 1:
        return x
    n_trim = (len(x) // tau) * tau
    return x[:n_trim].reshape(-1, tau).mean(axis=1)


def _chebyshev_match_count(windows: np.ndarray, r: float) -> int:
    """
    Count all ordered pairs (i, j) with i≠j whose Chebyshev distance ≤ r.

    windows : (N, d) array — each row is a flattened template.
    Returns an integer (double the number of unordered matches).
    """
    dists = cdist(windows.astype(np.float32), windows.astype(np.float32),
                  metric='chebyshev')
    matches = dists <= r
    np.fill_diagonal(matches, False)
    return int(matches.sum())


# ============================================================================
# LMC Statistical Complexity
# ============================================================================

def lmc_complexity(
    weights: np.ndarray,
    n_bins: Optional[int] = None,
) -> dict[str, float]:
    """
    Compute LMC statistical complexity C = H × D on a flattened weight array.

    Follows the convention validated by the research group (supervisor's code):

      bins = min(100, max(10, n // 5))   adaptive to sample size
      H    = -Σ p_i log₂(p_i)           Shannon entropy in bits
      D    =  Σ(p_i - 1/N)²             disequilibrium (L2 distance from uniform)
      C    = H × D                       unnormalized product

    Note: H and D are NOT normalized to [0,1].  The scale of C therefore
    depends on the number of bins and the weight distribution.  When comparing
    across layers or runs always use the same bin count.

    Returns a dict with keys:
        entropy       : Shannon entropy H (bits, log₂)
        disequilibrium: disequilibrium D (raw L2 from uniform)
        complexity    : LMC complexity C = H × D
    """
    flat = np.asarray(weights, dtype=np.float64).ravel()
    n = len(flat)
    if n_bins is None:
        n_bins = min(100, max(10, n // 5))

    counts, _ = np.histogram(flat, bins=n_bins)
    total = counts.sum()
    if total == 0:
        return dict(entropy=0.0, disequilibrium=0.0, complexity=0.0)

    p = counts / float(total)
    N = n_bins

    # Shannon entropy in bits (log₂) — supervisor's validated convention
    nz = p > 0
    H = float(-np.sum(p[nz] * np.log2(p[nz])))

    # Disequilibrium: L2 distance from the uniform distribution
    D = float(np.sum((p - 1.0 / N) ** 2))

    C = H * D

    return dict(entropy=H, disequilibrium=D, complexity=C)


# ============================================================================
# 1D Sample Entropy  (Richman & Moorman 2000)
# ============================================================================

def sample_entropy_1d(
    series: np.ndarray,
    m: int = 2,
    r: Optional[float] = None,
    r_scale: float = 0.2,
) -> float:
    """
    Compute 1D Sample Entropy following Richman & Moorman (2000).

    SampEn(m, r) = -ln(A / B)

    where:
        B = #{ordered pairs (i,j), i≠j, in [0,N-m-1]: Chebyshev_m(i,j) ≤ r}
        A = #{ordered pairs (i,j), i≠j, in [0,N-m-1]: Chebyshev_{m+1}(i,j) ≤ r}

    The template length for the (m+1) count uses the SAME set of N-m starting
    positions as for m, which is the standard Richman & Moorman convention.

    Parameters
    ----------
    series  : 1-D array of length ≥ m+2
    m       : embedding dimension (default 2)
    r       : absolute tolerance; if None, uses r_scale × SD(series, ddof=0)
    r_scale : fraction of SD used to derive r (default 0.2, supervisor's convention)

    Returns
    -------
    float — SampEn value (0 = perfectly regular, higher = more irregular).
            Returns np.inf when B > 0 but A = 0 (maximum irregularity).
            Returns np.nan when B = 0 (undefined).
    """
    x = np.asarray(series, dtype=np.float64).ravel()
    N = len(x)
    if N < m + 2:
        return np.nan

    if r is None:
        sd = float(np.std(x))   # ddof=0, matching supervisor's validated code
        if sd == 0.0:
            return 0.0   # constant series is perfectly regular
        r = r_scale * sd

    # Build template matrices via stride tricks
    # Shape (N-m, m) for length-m templates; same rows for length-(m+1)
    n_tpl = N - m   # number of valid starting positions

    def _template_matrix(length: int) -> np.ndarray:
        idx = np.arange(length)[None, :] + np.arange(n_tpl)[:, None]
        return x[idx]   # (n_tpl, length)

    B = _chebyshev_match_count(_template_matrix(m), r)
    A = _chebyshev_match_count(_template_matrix(m + 1), r)

    if B == 0:
        return np.nan
    if A == 0:
        return np.inf

    return float(-np.log(A / B))


# ============================================================================
# 1D Multiscale Sample Entropy  (Costa et al. 2002 / 2005)
# ============================================================================

def mse_1d(
    series: np.ndarray,
    max_scale: int = 10,
    m: int = 2,
    r_scale: float = 0.2,
) -> np.ndarray:
    """
    Compute 1D Multiscale Sample Entropy (MSE) following Costa et al. (2002/2005).

    At each scale τ (1 ≤ τ ≤ max_scale):
      1. Coarse-grain: average over non-overlapping τ-length blocks.
      2. Compute SampEn₁D(coarsened, m, r) with r fixed from the original series.

    The tolerance r = r_scale × SD(series) is computed ONCE from the original
    (τ=1) series and kept constant across all scales.

    Returns
    -------
    np.ndarray of shape (max_scale,) — SampEn at each scale τ = 1..max_scale.
    np.nan at a given scale indicates insufficient data after coarse-graining.
    """
    x = np.asarray(series, dtype=np.float64).ravel()

    sd = float(np.std(x))   # ddof=0, matching supervisor's validated code
    if sd == 0.0:
        return np.zeros(max_scale)
    r = r_scale * sd

    result = np.empty(max_scale)
    for tau in range(1, max_scale + 1):
        coarsened = _coarse_grain_1d(x, tau)
        result[tau - 1] = sample_entropy_1d(coarsened, m=m, r=r)

    return result

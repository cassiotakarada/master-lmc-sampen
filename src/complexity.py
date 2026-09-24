"""
src/complexity.py — Statistical complexity measures for NN weight analysis.

Implements:
  - LMC statistical complexity    (López-Ruiz, Mancini & Calbet, 1995)
  - 1D Sample Entropy             (Richman & Moorman, 2000; Pincus, 1991)
  - 2D Sample Entropy             (Proposal Eqs. 6-11; Silva et al., 2018)
  - 1D Multiscale Sample Entropy  (Costa, Goldberger & Peng, 2002 / 2005)
  - 2D Multiscale Sample Entropy  (Silva et al., 2018)

All functions operate on numpy arrays and are pure (no side effects).
Default parameters match the conventions validated by the research group:
  - SampEn: m=2, r = 0.2 × SD(series, ddof=0)   [supervisor's validated code]
  - MSE: max_scale=10 with r fixed from the original (τ=1) series
  - LMC: Shannon entropy in bits (log₂), unnormalized product H × D
         bins = min(100, max(10, n // 5))          [supervisor's validated code]

Design notes:
  - SampEn₂D applies to the weight matrix treated as a 2D spatial structure.
    For large matrices the matrix is subsampled to max_dim × max_dim before
    computing windows; the SD-based tolerance r is computed from the original
    (pre-subsampled) matrix.
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

def _subsample_2d(mat: np.ndarray, max_dim: int) -> np.ndarray:
    """Subsample a 2D matrix to at most max_dim × max_dim by strided indexing."""
    H, W = mat.shape
    sh = max(1, H // max_dim)
    sw = max(1, W // max_dim)
    return np.ascontiguousarray(mat[::sh, ::sw][:max_dim, :max_dim])


def _coarse_grain_1d(x: np.ndarray, tau: int) -> np.ndarray:
    """
    Coarse-grain a 1D series by averaging over non-overlapping tau-length blocks.
    Trailing samples that do not fill a complete block are discarded.
    """
    if tau == 1:
        return x
    n_trim = (len(x) // tau) * tau
    return x[:n_trim].reshape(-1, tau).mean(axis=1)


def _coarse_grain_2d(mat: np.ndarray, tau: int) -> np.ndarray:
    """
    Coarse-grain a 2D matrix by averaging over non-overlapping tau × tau blocks.
    Trailing rows/columns that do not complete a block are discarded.
    """
    if tau == 1:
        return mat
    H, W = mat.shape
    H_t = (H // tau) * tau
    W_t = (W // tau) * tau
    trimmed = mat[:H_t, :W_t]
    return trimmed.reshape(H_t // tau, tau, W_t // tau, tau).mean(axis=(1, 3))


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


def _extract_valid_windows(mat: np.ndarray, m: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract windows of size m×m AND (m+1)×(m+1) from the same set of valid
    starting positions (i,j) where both windows fit inside the matrix.

    Valid positions: 0 ≤ i ≤ H-m-1, 0 ≤ j ≤ W-m-1  → N = (H-m)(W-m) positions.

    Returns (wins_m, wins_m1):
      wins_m  : (N, m²) array of flattened m-windows
      wins_m1 : (N, (m+1)²) array of flattened (m+1)-windows
    """
    H, W = mat.shape
    n_rows = H - m          # number of valid row positions
    n_cols = W - m          # number of valid col positions
    if n_rows < 1 or n_cols < 1:
        empty_m = np.empty((0, m * m), dtype=np.float64)
        empty_m1 = np.empty((0, (m + 1) ** 2), dtype=np.float64)
        return empty_m, empty_m1

    # sliding_window_view: shape (H-m+1, W-m+1, m, m)
    # We need rows [0:H-m] and cols [0:W-m]
    wins_m_full = np.lib.stride_tricks.sliding_window_view(mat, (m, m))
    wins_m = wins_m_full[:n_rows, :n_cols].reshape(-1, m * m).astype(np.float64)

    # shape (H-m, W-m, m+1, m+1) — already the right number of rows/cols
    wins_m1_full = np.lib.stride_tricks.sliding_window_view(mat, (m + 1, m + 1))
    wins_m1 = wins_m1_full.reshape(-1, (m + 1) ** 2).astype(np.float64)

    return wins_m, wins_m1


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
# 2D Sample Entropy  (Proposal Eqs. 6-11; Silva et al. 2018)
# ============================================================================

def sample_entropy_2d(
    matrix: np.ndarray,
    m: int = 2,
    r: Optional[float] = None,
    r_scale: float = 0.2,
    max_dim: int = 64,
) -> float:
    """
    Compute 2D Sample Entropy following the research proposal (Eqs. 6-11).

    SampEn₂D(u, m, r) = -ln(U^{m+1}(r) / U^m(r))

    where U^m(r) and U^{m+1}(r) are the mean conditional match fractions for
    square windows of size m×m and (m+1)×(m+1), respectively.  The counting
    is restricted to positions (i,j) where BOTH window sizes fit:
        0 ≤ i ≤ H-m-1,  0 ≤ j ≤ W-m-1  →  N = (H-m)(W-m) positions.

    Distance metric: Chebyshev (max of element-wise absolute differences),
    matching the proposal's Eq. 11.

    Parameters
    ----------
    matrix  : 2-D weight matrix (any dtype; converted to float64 internally)
    m       : window size (default 2)
    r       : absolute tolerance; if None, r = r_scale × SD(matrix, ddof=0)
    r_scale : fraction of SD (default 0.2, matching supervisor's convention)
    max_dim : if either dimension of the matrix exceeds this, the matrix is
              subsampled to max_dim × max_dim BEFORE extracting windows.
              SD for r is computed from the ORIGINAL matrix.

    Returns
    -------
    float — SampEn₂D value.
            Returns 0.0 when the matrix is constant (SD = 0).
            Returns np.inf when U^m > 0 but U^{m+1} = 0.
            Returns np.nan when matrix is too small or U^m = 0.
    """
    mat = np.asarray(matrix, dtype=np.float64)
    if mat.ndim != 2:
        raise ValueError(f"Expected 2-D array, got shape {mat.shape}")

    # r is computed from the original (pre-subsampled) matrix
    if r is None:
        sd = float(np.std(mat))
        if sd == 0.0:
            return 0.0   # constant matrix → all windows identical → SampEn = 0
        r = r_scale * sd

    # Subsample for tractable window extraction
    mat_sub = _subsample_2d(mat, max_dim)
    H, W = mat_sub.shape

    if H < m + 1 or W < m + 1:
        return np.nan   # matrix too small for windows of size m and m+1

    wins_m, wins_m1 = _extract_valid_windows(mat_sub, m)
    N = wins_m.shape[0]
    if N < 2:
        return np.nan

    # Total ordered match counts (2 × unordered matches, factor cancels in ratio)
    B = _chebyshev_match_count(wins_m, r)
    A = _chebyshev_match_count(wins_m1, r)

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


# ============================================================================
# 2D Multiscale Sample Entropy  (Silva et al. 2018)
# ============================================================================

def mse_2d(
    matrix: np.ndarray,
    max_scale: int = 10,
    m: int = 2,
    r_scale: float = 0.2,
    max_dim: int = 64,
) -> np.ndarray:
    """
    Compute 2D Multiscale Sample Entropy (MSE₂D) following Silva et al. (2018).

    At each scale τ (1 ≤ τ ≤ max_scale):
      1. Coarse-grain: average over non-overlapping τ×τ blocks.
      2. Compute SampEn₂D(coarsened, m, r) with r fixed from the original matrix.

    The tolerance r = r_scale × SD(matrix) is computed ONCE from the original
    matrix and kept constant across all scales (analogous to Costa's convention).

    Parameters
    ----------
    matrix    : 2-D weight matrix
    max_scale : number of scales to compute (default 10)
    m         : window size for SampEn₂D (default 2)
    r_scale   : fraction of SD for tolerance (default 0.15)
    max_dim   : maximum spatial dimension after subsampling (passed to SampEn₂D)

    Returns
    -------
    np.ndarray of shape (max_scale,) — SampEn₂D at each scale τ = 1..max_scale.
    """
    mat = np.asarray(matrix, dtype=np.float64)
    if mat.ndim != 2:
        raise ValueError(f"Expected 2-D array, got shape {mat.shape}")

    sd = float(np.std(mat))
    if sd == 0.0:
        return np.zeros(max_scale)
    r = r_scale * sd

    result = np.empty(max_scale)
    for tau in range(1, max_scale + 1):
        coarsened = _coarse_grain_2d(mat, tau)
        # Pass precomputed r (no r_scale recomputation inside sample_entropy_2d)
        result[tau - 1] = sample_entropy_2d(coarsened, m=m, r=r, max_dim=max_dim)

    return result

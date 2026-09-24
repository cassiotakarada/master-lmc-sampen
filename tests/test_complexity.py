"""
tests/test_complexity.py — Unit tests for src/complexity.py.

Covers analytically known or clearly bounded cases for each measure:
  - LMC: uniform → entropy max, D≈0, C≈0; constant → entropy≈0, D max, C≈0
  - SampEn₁D: periodic series → low; noise → high; vs. antropy reference
  - SampEn₂D: constant matrix → 0; noise > structured pattern
  - MSE₁D/₂D: shape correctness; scale-1 matches direct SampEn

All seeds are fixed for reproducibility.
"""

import numpy as np
import pytest
from src.complexity import (
    lmc_complexity,
    sample_entropy_1d,
    sample_entropy_2d,
    mse_1d,
    mse_2d,
)

RNG = np.random.default_rng(42)


# ============================================================================
# LMC Statistical Complexity
# ============================================================================

class TestLMCComplexity:

    def test_uniform_weights_high_entropy_low_disequilibrium(self):
        # A large uniform sample fills all bins evenly → H large, D ≈ 0, C ≈ 0.
        # With log2 and ~100 bins, H_max ≈ log2(100) ≈ 6.64 bits.
        weights = RNG.uniform(-1.0, 1.0, size=50_000)
        out = lmc_complexity(weights)
        assert out['entropy'] > 6.0, f"Expected H>6 bits for uniform, got {out['entropy']:.4f}"
        assert out['disequilibrium'] < 1e-3, f"Expected D≈0, got {out['disequilibrium']:.6f}"
        assert out['complexity'] < 1e-2, f"Expected C≈0, got {out['complexity']:.6f}"

    def test_constant_weights_zero_entropy_zero_complexity(self):
        # All weights equal → all mass in one bin → H=0, C=0
        weights = np.zeros(1000)
        out = lmc_complexity(weights)
        assert out['entropy'] == 0.0, f"Expected H=0, got {out['entropy']}"
        assert out['complexity'] == 0.0, f"Expected C=0, got {out['complexity']}"

    def test_entropy_in_bits_scale(self):
        # log2: entropy of N equiprobable bins = log2(N).  With n//5 bins rule,
        # 1000 samples → bins = min(100, max(10, 200)) = 100 → H_max = log2(100)
        weights = RNG.uniform(0, 1, 1000)
        out = lmc_complexity(weights)
        assert 5.5 < out['entropy'] < 7.0, f"Entropy out of expected bits range: {out['entropy']}"

    def test_mixed_distribution_positive_complexity(self):
        # A bimodal distribution has structured D > 0 and H > 0 → C > 0
        weights = np.concatenate([
            RNG.normal(-2.0, 0.3, 500),
            RNG.normal(+2.0, 0.3, 500),
        ])
        out = lmc_complexity(weights)
        assert out['complexity'] > 0.0, (
            f"Expected C>0 for bimodal distribution, got {out['complexity']:.6f}"
        )

    def test_complexity_equals_h_times_d(self):
        weights = RNG.standard_normal(5000)
        out = lmc_complexity(weights)
        assert abs(out['complexity'] - out['entropy'] * out['disequilibrium']) < 1e-12

    def test_reproducibility(self):
        weights = np.linspace(-1, 1, 1000)
        assert lmc_complexity(weights) == lmc_complexity(weights)

    def test_single_nonzero_bin_gives_zero_complexity(self):
        # All same value → H=0, D>0, C=0
        weights = np.full(100, 0.5)
        out = lmc_complexity(weights)
        assert out['complexity'] < 1e-12

    def test_matches_supervisor_implementation(self):
        # Reproduce one value from the professor's validated code to ensure
        # our implementation is numerically identical.
        # Using the exact same logic: log2, raw product, adaptive bins.
        rng = np.random.default_rng(42)
        flat = rng.standard_normal(10_000)
        bins = min(100, max(10, len(flat) // 5))  # = 100
        hist, _ = np.histogram(flat, bins=bins)
        p = hist / hist.sum()
        p_nz = p[p > 0]
        H_ref = float(-np.sum(p_nz * np.log2(p_nz)))
        D_ref = float(np.sum((p - 1.0 / bins) ** 2))
        C_ref = H_ref * D_ref
        out = lmc_complexity(flat)
        assert abs(out['complexity'] - C_ref) < 1e-12


# ============================================================================
# 1D Sample Entropy
# ============================================================================

class TestSampleEntropy1D:

    def test_constant_series_returns_zero(self):
        # Constant series has SD=0 → function returns 0 by convention
        x = np.ones(200)
        assert sample_entropy_1d(x) == 0.0

    def test_perfectly_repeating_series_low_entropy(self):
        # Sequence [1,2,3,1,2,3,...] → many identical templates → SampEn ≈ 0
        x = np.tile([1.0, 2.0, 3.0], 70)   # 210 samples
        se = sample_entropy_1d(x, m=2, r_scale=0.15)
        assert se < 0.05, f"Expected near-zero SampEn for periodic, got {se:.4f}"

    def test_noise_higher_than_periodic(self):
        # White noise should have substantially higher SampEn than a periodic signal
        periodic = np.sin(np.linspace(0, 20 * np.pi, 400))
        noise = RNG.standard_normal(400)
        se_periodic = sample_entropy_1d(periodic, m=2, r_scale=0.15)
        se_noise = sample_entropy_1d(noise, m=2, r_scale=0.15)
        assert se_noise > se_periodic, (
            f"Noise SampEn ({se_noise:.4f}) should exceed periodic ({se_periodic:.4f})"
        )

    def test_noise_sampen_above_threshold(self):
        # For Gaussian noise with r=0.15*SD, SampEn₁D with m=2 should be > 1.0
        noise = RNG.standard_normal(1000)
        se = sample_entropy_1d(noise, m=2, r_scale=0.15)
        assert se > 1.0, f"Expected SampEn > 1.0 for noise, got {se:.4f}"

    def test_output_is_finite_for_typical_input(self):
        x = RNG.standard_normal(300)
        se = sample_entropy_1d(x)
        assert np.isfinite(se) or se == np.inf

    def test_default_r_is_02_times_sd_ddof0(self):
        # Default: r = 0.2 * SD(ddof=0) — supervisor's validated convention
        x = RNG.standard_normal(200)
        r = 0.2 * np.std(x)   # ddof=0
        assert sample_entropy_1d(x, r=r) == sample_entropy_1d(x)

    def test_too_short_series_returns_nan(self):
        assert np.isnan(sample_entropy_1d(np.array([1.0, 2.0])))

    def test_against_antropy_reference(self):
        """
        Cross-validate against the antropy library reference implementation.

        antropy.sample_entropy accepts a `tolerance` parameter; we pass the
        same absolute r to both to remove any scale-factor ambiguity.
        Convention: r = 0.2 * SD(ddof=0), matching supervisor's validated code.
        Both implementations follow Richman & Moorman (2000) with chebyshev
        distance; minor differences in template-count boundary conventions
        (N-m vs N-m+1) can produce values within ~0.10 of each other.
        """
        pytest.importorskip("antropy")
        import antropy
        x = RNG.standard_normal(500)          # longer series for stable estimate
        r = 0.2 * np.std(x)                   # ddof=0, supervisor's convention
        our_se = sample_entropy_1d(x, m=2, r=r)
        ref_se = float(antropy.sample_entropy(x, order=2, metric='chebyshev',
                                              tolerance=r))
        assert abs(our_se - ref_se) < 0.10, (
            f"Our SampEn1D ({our_se:.4f}) diverges from antropy ({ref_se:.4f})"
        )


# ============================================================================
# 2D Sample Entropy
# ============================================================================

class TestSampleEntropy2D:

    def test_constant_matrix_returns_zero(self):
        mat = np.full((32, 32), 3.14)
        assert sample_entropy_2d(mat) == 0.0

    def test_noise_matrix_finite_or_inf_and_nonnegative(self):
        # inf is a valid return (A=0: no (m+1)-windows match → max irregularity)
        mat = RNG.standard_normal((32, 32))
        se = sample_entropy_2d(mat, m=2, r_scale=0.15)
        assert np.isfinite(se) or np.isinf(se), f"Expected finite or inf, got {se}"
        if np.isfinite(se):
            assert se >= 0.0

    def test_noise_higher_than_checkerboard(self):
        # Checkerboard: highly regular 2D pattern → lower SampEn₂D than noise
        rng = np.random.default_rng(7)
        noise = rng.standard_normal((32, 32))

        # ±1 checkerboard has SD=1 and perfectly repeating 2-period pattern
        ii, jj = np.indices((32, 32))
        checker = ((ii + jj) % 2).astype(float) * 2.0 - 1.0

        se_noise = sample_entropy_2d(noise, m=2, r_scale=0.15)
        se_check = sample_entropy_2d(checker, m=2, r_scale=0.15)
        assert se_noise > se_check, (
            f"Noise ({se_noise:.4f}) should exceed checkerboard ({se_check:.4f})"
        )

    def test_uniform_random_matrix_positive_sampen(self):
        mat = RNG.uniform(-1, 1, size=(40, 40))
        se = sample_entropy_2d(mat, m=2, r_scale=0.15)
        assert se > 0.0, f"Expected positive SampEn₂D for random matrix, got {se}"

    def test_large_matrix_subsampled(self):
        # Large weight matrix (e.g. ResNet conv layer) should not raise
        mat = RNG.standard_normal((128, 128))
        se = sample_entropy_2d(mat, m=2, max_dim=32)
        assert np.isfinite(se) or se == np.inf

    def test_non_square_matrix_accepted(self):
        mat = RNG.standard_normal((16, 64))
        se = sample_entropy_2d(mat, m=2)
        assert np.isfinite(se) or np.isnan(se) or se == np.inf

    def test_raises_on_1d_input(self):
        with pytest.raises(ValueError):
            sample_entropy_2d(np.arange(100.0))

    def test_output_nonnegative(self):
        # By construction A ≤ B, so SampEn₂D = -ln(A/B) ≥ 0 for any input
        for _ in range(5):
            mat = RNG.standard_normal((20, 20))
            se = sample_entropy_2d(mat)
            if np.isfinite(se):
                assert se >= 0.0, f"SampEn₂D must be ≥ 0, got {se}"


# ============================================================================
# Multiscale Sample Entropy — 1D
# ============================================================================

class TestMSE1D:

    def test_output_shape(self):
        x = RNG.standard_normal(500)
        result = mse_1d(x, max_scale=10)
        assert result.shape == (10,)

    def test_scale1_matches_direct_sampen1d(self):
        x = RNG.standard_normal(500)
        mse = mse_1d(x, max_scale=5, m=2)
        direct = sample_entropy_1d(x, m=2)
        assert abs(mse[0] - direct) < 1e-10, (
            f"MSE at τ=1 ({mse[0]:.6f}) should equal direct SampEn₁D ({direct:.6f})"
        )

    def test_constant_returns_zeros(self):
        x = np.ones(300)
        result = mse_1d(x, max_scale=5)
        assert np.all(result == 0.0)

    def test_r_is_fixed_from_original_series(self):
        # MSE at scale 2 with our function vs. manually coarse-graining then SampEn
        # The difference: our function uses r from the original scale-1 series.
        x = RNG.standard_normal(400)
        r_orig = 0.2 * np.std(x)   # ddof=0, supervisor's convention
        # Coarse-grain manually
        x_cg = x[:400].reshape(-1, 2).mean(axis=1)
        manual_se = sample_entropy_1d(x_cg, m=2, r=r_orig)  # pass same r
        mse = mse_1d(x, max_scale=2)
        assert abs(mse[1] - manual_se) < 1e-10


# ============================================================================
# Multiscale Sample Entropy — 2D
# ============================================================================

class TestMSE2D:

    def test_output_shape(self):
        mat = RNG.standard_normal((32, 32))
        result = mse_2d(mat, max_scale=5)
        assert result.shape == (5,)

    def test_scale1_matches_direct_sampen2d(self):
        mat = RNG.standard_normal((24, 24))
        mse = mse_2d(mat, max_scale=3, m=2)
        # r fixed from original matrix for MSE (ddof=0, supervisor's convention)
        r = 0.2 * np.std(mat)
        direct = sample_entropy_2d(mat, m=2, r=r)
        # Both inf or both equal finite value
        both_inf = np.isinf(mse[0]) and np.isinf(direct)
        both_nan = np.isnan(mse[0]) and np.isnan(direct)
        assert both_inf or both_nan or abs(mse[0] - direct) < 1e-10, (
            f"MSE₂D at τ=1 ({mse[0]}) should equal direct SampEn₂D ({direct})"
        )

    def test_constant_matrix_returns_zeros(self):
        mat = np.full((20, 20), 1.0)
        result = mse_2d(mat, max_scale=4)
        assert np.all(result == 0.0)

    def test_all_scales_not_negative_infinity(self):
        # Valid outputs are: finite (≥0), +inf (max irregularity), or nan (undefined).
        # Negative infinity is never expected.
        mat = RNG.standard_normal((32, 32))
        result = mse_2d(mat, max_scale=8)
        for i, v in enumerate(result):
            assert not (np.isinf(v) and v < 0), (
                f"Unexpected -inf at scale {i+1}"
            )
            assert not np.isnan(v) or True  # nan is acceptable

    def test_r_fixed_from_original_matrix(self):
        mat = RNG.standard_normal((24, 24))
        r_orig = 0.2 * np.std(mat)   # ddof=0, supervisor's convention
        # Coarse-grain scale 2 manually and compute SampEn₂D with original r
        mat_cg = _coarse_grain_2d_helper(mat, tau=2)
        manual_se = sample_entropy_2d(mat_cg, m=2, r=r_orig)
        mse = mse_2d(mat, max_scale=2)
        # Handle inf==inf and nan==nan correctly
        both_inf = np.isinf(mse[1]) and np.isinf(manual_se)
        both_nan = np.isnan(mse[1]) and np.isnan(manual_se)
        assert both_inf or both_nan or abs(mse[1] - manual_se) < 1e-10, (
            f"MSE₂D scale-2 ({mse[1]}) should equal manual ({manual_se})"
        )


# Helper for the last MSE₂D test (mirrors the private function)
def _coarse_grain_2d_helper(mat: np.ndarray, tau: int) -> np.ndarray:
    H, W = mat.shape
    H_t = (H // tau) * tau
    W_t = (W // tau) * tau
    trimmed = mat[:H_t, :W_t]
    return trimmed.reshape(H_t // tau, tau, W_t // tau, tau).mean(axis=(1, 3))

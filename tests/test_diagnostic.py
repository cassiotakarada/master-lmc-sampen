"""
tests/test_diagnostic.py — Unit tests for scripts/diagnostic.py.

Imports the diagnostic module via sys.path (the same trick the script itself
uses) so no package restructuring is needed.

Covers:
  - sanity_checks: passes with valid data, fails with empty/bad data
  - make_plots: creates the PNG file, handles edge cases
  - main(): end-to-end with a mock train_run (no MedMNIST download)
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Allow 'import diagnostic' without installing as a package
_SCRIPTS = Path(__file__).parent.parent / 'scripts'
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import diagnostic as diag


# ============================================================================
# Shared helpers
# ============================================================================

def _make_metrics(n_steps: int = 5, groups=('conv', 'dense')) -> pd.DataFrame:
    """Synthetic metrics DataFrame that mirrors train_run output."""
    rows = []
    rng = np.random.default_rng(0)
    for step in range(1, n_steps + 1):
        for grp in groups:
            rows.append({
                'run_id':             'diagnostic',
                'step':               step,
                'epoch':              step / n_steps,
                'group':              grp,
                'n_flat_samples':     200,
                'n_units':            8,
                'lmc_entropy':        rng.uniform(3.0, 5.0),
                'lmc_disequilibrium': rng.uniform(0.001, 0.01),
                'lmc_complexity':     rng.uniform(0.01, 0.05),
                'sampen_1d':          rng.uniform(1.5, 2.5),
                'l2_mean':            rng.uniform(0.1, 1.0),
                'l2_std':             rng.uniform(0.01, 0.2),
            })
    return pd.DataFrame(rows)


def _make_history(n_epochs: int = 2) -> pd.DataFrame:
    rows = []
    for ep in range(1, n_epochs + 1):
        rows.append({'step': ep * 10, 'epoch': ep, 'phase': 'train',
                     'loss': 0.5, 'acc': 0.7 + ep * 0.1})
        rows.append({'step': ep * 10, 'epoch': ep, 'phase': 'val',
                     'loss': 0.6, 'acc': 0.65 + ep * 0.1})
    return pd.DataFrame(rows)


# ============================================================================
# sanity_checks
# ============================================================================

class TestSanityChecks:

    def test_passes_with_valid_data(self, capsys):
        ok = diag.sanity_checks(_make_metrics(), _make_history())
        assert ok is True

    def test_fails_with_empty_metrics(self, capsys):
        ok = diag.sanity_checks(pd.DataFrame(), _make_history())
        assert ok is False

    def test_fails_with_empty_history(self, capsys):
        ok = diag.sanity_checks(_make_metrics(), pd.DataFrame())
        assert ok is False

    def test_fails_when_lmc_not_positive(self, capsys):
        m = _make_metrics()
        m.loc[0, 'lmc_complexity'] = -1.0
        ok = diag.sanity_checks(m, _make_history())
        assert ok is False

    def test_fails_when_l2_negative(self, capsys):
        m = _make_metrics()
        m.loc[0, 'l2_mean'] = -0.5
        ok = diag.sanity_checks(m, _make_history())
        assert ok is False

    def test_fails_when_val_acc_out_of_range(self, capsys):
        h = _make_history()
        h.loc[h['phase'] == 'val', 'acc'] = 1.5   # invalid
        ok = diag.sanity_checks(_make_metrics(), h)
        assert ok is False

    def test_accepts_some_nan_sampen(self, capsys):
        m = _make_metrics(n_steps=10)
        # NaN in a few rows is acceptable (≥ 50% must be finite)
        m.loc[m.index[:3], 'sampen_1d'] = float('nan')
        ok = diag.sanity_checks(m, _make_history())
        assert ok is True

    def test_fails_when_too_many_nan_sampen(self, capsys):
        m = _make_metrics(n_steps=4, groups=('conv',))
        m['sampen_1d'] = float('nan')   # 100% NaN → < 50% finite
        ok = diag.sanity_checks(m, _make_history())
        assert ok is False

    def test_fails_when_groups_missing(self, capsys):
        # Only 'bn_gamma' present — 'conv' and 'dense' are missing
        m = _make_metrics(groups=('bn_gamma',))
        ok = diag.sanity_checks(m, _make_history())
        assert ok is False


# ============================================================================
# make_plots
# ============================================================================

class TestMakePlots:

    def test_creates_png_file(self, tmp_path):
        plot_dir = tmp_path / 'plots'
        result = diag.make_plots(_make_metrics(), _make_history(), plot_dir)
        assert result is not None
        assert result.exists()
        assert result.suffix == '.png'

    def test_returns_none_for_empty_metrics(self, tmp_path):
        result = diag.make_plots(pd.DataFrame(), _make_history(),
                                  tmp_path / 'plots')
        assert result is None

    def test_handles_all_nan_sampen(self, tmp_path):
        m = _make_metrics()
        m['sampen_1d'] = float('nan')
        plot_dir = tmp_path / 'plots'
        result = diag.make_plots(m, _make_history(), plot_dir)
        assert result is not None   # should still produce a plot (just empty SampEn line)

    def test_handles_empty_history(self, tmp_path):
        result = diag.make_plots(_make_metrics(), pd.DataFrame(),
                                  tmp_path / 'plots')
        assert result is not None


# ============================================================================
# main() — end-to-end with mock train_run
# ============================================================================

class TestMain:
    """
    Monkeypatch diag.train_run so main() can run without MedMNIST or GPU.
    The mock writes synthetic CSVs to output_dir and returns the expected dict.
    """

    def _setup_mock_run(self, tmp_path: Path):
        """Write synthetic CSV outputs and return a mock train_fn."""
        metrics = _make_metrics()
        history = _make_history()

        def _mock(config, output_dir, resume=True, _loaders=None):
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            m_path = out / 'metrics.csv'
            h_path = out / 'history.csv'
            metrics.to_csv(m_path, index=False)
            history.to_csv(h_path, index=False)
            (out / 'state.json').write_text(json.dumps({
                'last_completed_epoch': config['epochs'],
                'last_step': 10,
                'processed_steps': [],
            }))
            return {
                'run_id':         config['run_id'],
                'final_val_acc':  0.85,
                'final_val_loss': 0.35,
                'metrics_path':   str(m_path),
                'history_path':   str(h_path),
            }

        return _mock

    def test_exits_0_on_success(self, tmp_path, monkeypatch):
        monkeypatch.setattr(diag, 'train_run', self._setup_mock_run(tmp_path))
        monkeypatch.chdir(tmp_path)   # so 'results/' is created in tmp_path

        with pytest.raises(SystemExit) as exc_info:
            diag.main(['--data-root', str(tmp_path / 'data')])

        assert exc_info.value.code == 0

    def test_plot_created(self, tmp_path, monkeypatch):
        monkeypatch.setattr(diag, 'train_run', self._setup_mock_run(tmp_path))
        monkeypatch.chdir(tmp_path)

        with pytest.raises(SystemExit):
            diag.main(['--data-root', str(tmp_path / 'data')])

        plot_candidates = list(tmp_path.rglob('diagnostic.png'))
        assert len(plot_candidates) >= 1

    def test_exits_1_when_train_fails(self, tmp_path, monkeypatch):
        def _failing(*a, **kw):
            raise RuntimeError("Simulated GPU OOM")

        monkeypatch.setattr(diag, 'train_run', _failing)
        monkeypatch.chdir(tmp_path)

        with pytest.raises(SystemExit) as exc_info:
            diag.main(['--data-root', str(tmp_path / 'data')])

        assert exc_info.value.code == 1

    def test_force_flag_passed_as_no_resume(self, tmp_path, monkeypatch):
        calls = []

        def _recording(config, output_dir, resume=True, _loaders=None):
            calls.append({'resume': resume})
            return self._setup_mock_run(tmp_path)(config, output_dir, resume)

        monkeypatch.setattr(diag, 'train_run', _recording)
        monkeypatch.chdir(tmp_path)

        with pytest.raises(SystemExit):
            diag.main(['--force', '--data-root', str(tmp_path / 'data')])

        assert calls[0]['resume'] is False

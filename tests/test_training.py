"""
tests/test_training.py — Unit and integration tests for src/training.py.

Integration tests bypass make_loaders via the _loaders parameter so no
MedMNIST download is required.  A tiny synthetic DataLoader is used instead.
All tests run on CPU and complete in a few seconds.
"""

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.training import (
    _accuracy,
    checkpoint_schedule,
    evaluate,
    extract_metrics_rows,
    make_model,
    train_run,
)
from src.weights import WeightSnapshot


# ============================================================================
# Shared synthetic DataLoader factory
# ============================================================================

def _synthetic_loaders(
    n_train: int = 32,
    n_val:   int = 16,
    n_classes: int = 4,
    image_size: int = 16,
    batch_size: int = 8,
) -> tuple[DataLoader, DataLoader, int]:
    """
    Build tiny in-memory DataLoaders that look like MedMNIST loaders
    (batch of (images, labels) with labels shaped (N, 1)).
    """
    rng = np.random.default_rng(0)
    def _ds(n):
        imgs = torch.from_numpy(
            rng.standard_normal((n, 3, image_size, image_size)).astype(np.float32)
        )
        labs = torch.from_numpy(
            rng.integers(0, n_classes, size=(n, 1)).astype(np.int64)
        )
        return TensorDataset(imgs, labs)

    train_dl = DataLoader(_ds(n_train), batch_size=batch_size, shuffle=True)
    val_dl   = DataLoader(_ds(n_val),   batch_size=batch_size, shuffle=False)
    return train_dl, val_dl, n_classes


# ============================================================================
# checkpoint_schedule
# ============================================================================

class TestCheckpointSchedule:

    def test_step_1_always_included(self):
        steps = checkpoint_schedule(total_steps=500, n_checkpoints=10)
        assert 1 in steps

    def test_length_at_most_n_checkpoints(self):
        steps = checkpoint_schedule(total_steps=1000, n_checkpoints=50)
        assert len(steps) <= 50

    def test_all_steps_within_bounds(self):
        total = 200
        steps = checkpoint_schedule(total_steps=total, n_checkpoints=20)
        assert all(1 <= s <= total for s in steps)

    def test_all_steps_included_when_total_le_n(self):
        steps = checkpoint_schedule(total_steps=10, n_checkpoints=100)
        assert steps == set(range(1, 11))

    def test_returns_set_of_ints(self):
        steps = checkpoint_schedule(total_steps=500, n_checkpoints=30)
        assert isinstance(steps, set)
        assert all(isinstance(s, int) for s in steps)

    def test_large_run_has_dense_early_steps(self):
        # Log-spacing guarantees several checkpoints in the first 1% of training
        total = 10_000
        steps = checkpoint_schedule(total_steps=total, n_checkpoints=100)
        early = {s for s in steps if s <= total // 100}
        assert len(early) >= 5, f"Expected ≥5 early checkpoints, got {early}"


# ============================================================================
# make_model
# ============================================================================

class TestMakeModel:

    def test_fc_output_dim(self):
        model = make_model(n_classes=7)
        assert model.fc.out_features == 7

    def test_forward_pass_shape(self):
        model = make_model(n_classes=4)
        model.eval()
        with torch.no_grad():
            out = model(torch.randn(2, 3, 64, 64))
        assert out.shape == (2, 4)

    def test_forward_pass_small_images(self):
        # AdaptiveAvgPool2d handles any spatial size ≥ 1×1
        model = make_model(n_classes=3)
        model.eval()
        with torch.no_grad():
            out = model(torch.randn(1, 3, 16, 16))
        assert out.shape == (1, 3)


# ============================================================================
# _accuracy
# ============================================================================

class TestAccuracy:

    def test_all_correct(self):
        logits = torch.tensor([[10.0, 0.0], [0.0, 10.0]])
        labels = torch.tensor([[0], [1]])
        assert _accuracy(logits, labels) == pytest.approx(1.0)

    def test_all_wrong(self):
        logits = torch.tensor([[10.0, 0.0], [10.0, 0.0]])
        labels = torch.tensor([[1], [1]])
        assert _accuracy(logits, labels) == pytest.approx(0.0)

    def test_half_correct(self):
        # [10,0]→pred 0, [0,10]→pred 1, [10,0]→pred 0, [0,10]→pred 1
        logits = torch.tensor([[10.0, 0.0], [0.0, 10.0],
                                [10.0, 0.0], [0.0, 10.0]])
        labels = torch.tensor([[0], [0], [1], [1]])   # items 0,3 correct → 0.5
        assert _accuracy(logits, labels) == pytest.approx(0.5)


# ============================================================================
# extract_metrics_rows
# ============================================================================

class TestExtractMetricsRows:

    def _make_snap(self):
        snap = WeightSnapshot(step=5, epoch=0.5, run_id='x')
        rng = np.random.default_rng(0)
        for grp in ('conv', 'dense'):
            snap.flat_weights[grp] = rng.standard_normal(500).astype(np.float64)
            snap.l2_norms[grp]     = np.abs(rng.standard_normal(16).astype(np.float64))
        return snap

    def test_one_row_per_group(self):
        snap = self._make_snap()
        rows = extract_metrics_rows(snap, run_id='r1')
        assert len(rows) == len(snap.groups())

    def test_required_keys_present(self):
        from src.training import METRICS_HEADER
        snap = self._make_snap()
        for row in extract_metrics_rows(snap, run_id='r1'):
            for key in METRICS_HEADER:
                assert key in row, f"Missing key '{key}' in row"

    def test_lmc_complexity_equals_h_times_d(self):
        snap = self._make_snap()
        for row in extract_metrics_rows(snap, run_id='r1'):
            h = row['lmc_entropy']
            d = row['lmc_disequilibrium']
            c = row['lmc_complexity']
            assert abs(c - h * d) < 1e-12

    def test_l2_mean_nonnegative(self):
        snap = self._make_snap()
        for row in extract_metrics_rows(snap, run_id='r1'):
            assert row['l2_mean'] >= 0.0

    def test_run_id_propagated(self):
        snap = self._make_snap()
        for row in extract_metrics_rows(snap, run_id='my_run'):
            assert row['run_id'] == 'my_run'


# ============================================================================
# train_run — integration tests with synthetic data
# ============================================================================

class TestTrainRun:

    BASE_CONFIG = {
        'run_id':          'test_run',
        'dataset':         'PathMNIST',
        'epochs':          2,
        'seed':            0,
        'lr':              0.01,
        'momentum':        0.9,
        'weight_decay':    0.0,
        'n_checkpoints':   5,
        'max_flat_sample': 200,
        'num_workers':     0,
    }

    def _run(self, tmp_path, config_override=None, resume=False):
        cfg = {**self.BASE_CONFIG, **(config_override or {})}
        loaders = _synthetic_loaders(
            n_train=32, n_val=16, n_classes=4, image_size=16, batch_size=8
        )
        return train_run(cfg, output_dir=tmp_path, resume=resume, _loaders=loaders)

    def test_returns_required_keys(self, tmp_path):
        result = self._run(tmp_path)
        for key in ('run_id', 'final_val_acc', 'final_val_loss',
                    'metrics_path', 'history_path'):
            assert key in result

    def test_metrics_csv_created(self, tmp_path):
        self._run(tmp_path)
        assert (tmp_path / 'metrics.csv').exists()

    def test_history_csv_created(self, tmp_path):
        self._run(tmp_path)
        assert (tmp_path / 'history.csv').exists()

    def test_config_json_created(self, tmp_path):
        self._run(tmp_path)
        assert (tmp_path / 'config.json').exists()

    def test_state_json_created(self, tmp_path):
        self._run(tmp_path)
        assert (tmp_path / 'state.json').exists()

    def test_state_json_last_completed_epoch(self, tmp_path):
        self._run(tmp_path)
        state = json.loads((tmp_path / 'state.json').read_text())
        assert state['last_completed_epoch'] == 2

    def test_history_csv_has_two_rows_per_epoch(self, tmp_path):
        self._run(tmp_path)
        with open(tmp_path / 'history.csv') as f:
            rows = list(csv.DictReader(f))
        # 2 epochs × 2 phases (train + val) = 4 rows
        assert len(rows) == 4

    def test_history_phases_correct(self, tmp_path):
        self._run(tmp_path)
        with open(tmp_path / 'history.csv') as f:
            phases = [r['phase'] for r in csv.DictReader(f)]
        assert set(phases) == {'train', 'val'}

    def test_metrics_csv_groups_subset_of_valid(self, tmp_path):
        from src.weights import GROUPS
        self._run(tmp_path)
        with open(tmp_path / 'metrics.csv') as f:
            groups = {r['group'] for r in csv.DictReader(f)}
        # bias group is excluded (include_bias_group=False)
        assert groups.issubset(set(GROUPS) - {'bias'})

    def test_metrics_csv_step_monotonic(self, tmp_path):
        self._run(tmp_path)
        with open(tmp_path / 'metrics.csv') as f:
            steps = [int(r['step']) for r in csv.DictReader(f)]
        # Steps may repeat (one per group) but overall should be non-decreasing
        assert steps == sorted(steps)

    def test_resume_skips_completed_run(self, tmp_path):
        # Run once to completion, then call again with resume=True
        self._run(tmp_path, resume=False)
        mtime_before = (tmp_path / 'metrics.csv').stat().st_mtime

        result = self._run(tmp_path, resume=True)
        mtime_after = (tmp_path / 'metrics.csv').stat().st_mtime

        # Already complete → skipped → metrics.csv not touched
        assert result['final_val_acc'] is None
        assert mtime_before == mtime_after

    def test_resume_checkpoint_file_exists(self, tmp_path):
        self._run(tmp_path)
        ckpt_files = list((tmp_path / 'checkpoints').glob('resume_epoch_*.pt'))
        assert len(ckpt_files) == 1   # only last epoch kept

    def test_run_id_in_metrics(self, tmp_path):
        self._run(tmp_path)
        with open(tmp_path / 'metrics.csv') as f:
            run_ids = {r['run_id'] for r in csv.DictReader(f)}
        assert run_ids == {'test_run'}

    def test_val_acc_between_0_and_1(self, tmp_path):
        result = self._run(tmp_path)
        assert 0.0 <= result['final_val_acc'] <= 1.0

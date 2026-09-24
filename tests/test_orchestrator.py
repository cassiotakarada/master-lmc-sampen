"""
tests/test_orchestrator.py — Unit tests for src/orchestrator.py.

All tests use temporary directories and a mock _train_fn so no actual
training or MedMNIST download occurs.

Covers:
  - normalise_dataset: casing normalisation
  - make_run_id: naming convention
  - load_grid / _expand: YAML → flat run list
  - status tracking: load/save/get/set
  - append_result: CSV output
  - maybe_trigger_writer: correct trigger count, skips missing writer
  - orchestrate: integration with mock train_fn
"""

import csv
import json
import textwrap
from pathlib import Path

import pytest

from src.orchestrator import (
    _RESULTS_HEADER,
    append_result,
    get_run_status,
    load_grid,
    load_status,
    make_run_id,
    maybe_trigger_writer,
    normalise_dataset,
    orchestrate,
    save_status,
    set_run_status,
)


# ============================================================================
# Shared helpers
# ============================================================================

def _write_yaml(tmp_path: Path, content: str) -> Path:
    p = tmp_path / 'grid.yaml'
    p.write_text(textwrap.dedent(content))
    return p


def _mock_train_fn(config, output_dir, resume=True, _loaders=None):
    """Minimal mock that satisfies orchestrate's contract."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'state.json').write_text(json.dumps({
        'last_completed_epoch': config['epochs'],
        'last_step': 10,
        'processed_steps': [],
    }))
    return {
        'run_id':         config['run_id'],
        'final_val_acc':  0.85,
        'final_val_loss': 0.40,
        'metrics_path':   str(out / 'metrics.csv'),
        'history_path':   str(out / 'history.csv'),
    }


# ============================================================================
# normalise_dataset
# ============================================================================

class TestNormaliseDataset:

    def test_lowercase_pathmnist(self):
        assert normalise_dataset('pathmnist') == 'PathMNIST'

    def test_uppercase_passthrough(self):
        assert normalise_dataset('PathMNIST') == 'PathMNIST'

    def test_organamnist(self):
        assert normalise_dataset('organamnist') == 'OrganAMNIST'

    def test_bloodmnist(self):
        assert normalise_dataset('bloodmnist') == 'BloodMNIST'

    def test_unknown_returned_as_is(self):
        assert normalise_dataset('CustomMNIST') == 'CustomMNIST'


# ============================================================================
# make_run_id
# ============================================================================

class TestMakeRunId:

    def test_format(self):
        assert make_run_id('PathMNIST', 30, 42) == 'PathMNIST_30ep_seed42'

    def test_different_seeds_distinct(self):
        r1 = make_run_id('PathMNIST', 30, 42)
        r2 = make_run_id('PathMNIST', 30, 123)
        assert r1 != r2

    def test_different_epochs_distinct(self):
        r1 = make_run_id('PathMNIST', 30, 42)
        r2 = make_run_id('PathMNIST', 50, 42)
        assert r1 != r2

    def test_contains_dataset_name(self):
        assert 'PathMNIST' in make_run_id('PathMNIST', 50, 0)


# ============================================================================
# load_grid / _expand
# ============================================================================

class TestLoadGrid:

    def test_basic_expansion(self, tmp_path):
        yaml_path = _write_yaml(tmp_path, """
            defaults:
              lr: 0.001
              batch_size: 64
            experiments:
              - dataset: PathMNIST
                epochs: 30
                seeds: [42, 123]
        """)
        _, runs = load_grid(yaml_path)
        assert len(runs) == 2

    def test_multiple_epoch_groups(self, tmp_path):
        yaml_path = _write_yaml(tmp_path, """
            defaults:
              lr: 0.001
            experiments:
              - dataset: PathMNIST
                epochs: 30
                seeds: [1, 2]
              - dataset: PathMNIST
                epochs: 50
                seeds: [1]
        """)
        _, runs = load_grid(yaml_path)
        assert len(runs) == 3

    def test_defaults_merged_into_runs(self, tmp_path):
        yaml_path = _write_yaml(tmp_path, """
            defaults:
              lr: 0.005
              batch_size: 32
            experiments:
              - dataset: PathMNIST
                epochs: 10
                seeds: [7]
        """)
        _, runs = load_grid(yaml_path)
        assert runs[0]['lr'] == 0.005
        assert runs[0]['batch_size'] == 32

    def test_run_id_populated(self, tmp_path):
        yaml_path = _write_yaml(tmp_path, """
            defaults: {}
            experiments:
              - dataset: PathMNIST
                epochs: 20
                seeds: [42]
        """)
        _, runs = load_grid(yaml_path)
        assert runs[0]['run_id'] == 'PathMNIST_20ep_seed42'

    def test_dataset_normalised_to_pascal_case(self, tmp_path):
        yaml_path = _write_yaml(tmp_path, """
            defaults: {}
            experiments:
              - dataset: pathmnist
                epochs: 10
                seeds: [1]
        """)
        _, runs = load_grid(yaml_path)
        assert runs[0]['dataset'] == 'PathMNIST'

    def test_empty_experiments(self, tmp_path):
        yaml_path = _write_yaml(tmp_path, """
            defaults: {}
            experiments: []
        """)
        _, runs = load_grid(yaml_path)
        assert runs == []

    def test_returns_defaults_dict(self, tmp_path):
        yaml_path = _write_yaml(tmp_path, """
            defaults:
              writer_trigger_every: 3
            experiments: []
        """)
        defaults, _ = load_grid(yaml_path)
        assert defaults['writer_trigger_every'] == 3


# ============================================================================
# Status tracking
# ============================================================================

class TestStatusTracking:

    def test_status_missing_returns_pending(self, tmp_path):
        assert get_run_status('nonexistent', tmp_path) == 'pending'

    def test_set_and_get_status(self, tmp_path):
        set_run_status('run_A', 'completed', tmp_path)
        assert get_run_status('run_A', tmp_path) == 'completed'

    def test_multiple_runs_tracked_independently(self, tmp_path):
        set_run_status('run_A', 'completed', tmp_path)
        set_run_status('run_B', 'failed', tmp_path)
        assert get_run_status('run_A', tmp_path) == 'completed'
        assert get_run_status('run_B', tmp_path) == 'failed'

    def test_status_persisted_to_json(self, tmp_path):
        set_run_status('run_X', 'running', tmp_path)
        data = json.loads((tmp_path / 'status.json').read_text())
        assert data['run_X'] == 'running'

    def test_status_overwrites_previous(self, tmp_path):
        set_run_status('run_A', 'running', tmp_path)
        set_run_status('run_A', 'completed', tmp_path)
        assert get_run_status('run_A', tmp_path) == 'completed'

    def test_load_status_empty_file(self, tmp_path):
        assert load_status(tmp_path) == {}

    def test_save_and_load_round_trip(self, tmp_path):
        original = {'r1': 'completed', 'r2': 'failed'}
        save_status(original, tmp_path)
        assert load_status(tmp_path) == original


# ============================================================================
# append_result
# ============================================================================

class TestAppendResult:

    def _result(self, run_id='r1'):
        return {
            'run_id': run_id,
            'final_val_acc': 0.9,
            'final_val_loss': 0.1,
            'metrics_path': '/tmp/metrics.csv',
            'history_path': '/tmp/history.csv',
        }

    def _cfg(self, dataset='PathMNIST', epochs=30, seed=42):
        return {'dataset': dataset, 'epochs': epochs, 'seed': seed}

    def test_creates_csv_on_first_call(self, tmp_path):
        append_result(self._result(), self._cfg(), tmp_path)
        assert (tmp_path / 'results.csv').exists()

    def test_csv_has_header_and_one_row(self, tmp_path):
        append_result(self._result(), self._cfg(), tmp_path)
        with open(tmp_path / 'results.csv') as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1

    def test_correct_columns(self, tmp_path):
        append_result(self._result('r2'), self._cfg(), tmp_path)
        with open(tmp_path / 'results.csv') as f:
            row = list(csv.DictReader(f))[0]
        for col in _RESULTS_HEADER:
            assert col in row

    def test_multiple_rows_accumulated(self, tmp_path):
        for i in range(3):
            append_result(self._result(f'run_{i}'), self._cfg(), tmp_path)
        with open(tmp_path / 'results.csv') as f:
            assert len(list(csv.DictReader(f))) == 3


# ============================================================================
# maybe_trigger_writer
# ============================================================================

class TestMaybeTriggerWriter:

    def test_does_not_trigger_on_non_multiple(self, tmp_path):
        # trigger_every=5, n_completed=3 → no trigger
        # Since writer.py doesn't exist, the function is a no-op anyway,
        # but we verify it doesn't raise.
        maybe_trigger_writer(tmp_path, n_completed=3, trigger_every=5)

    def test_skips_when_writer_missing(self, tmp_path):
        # n_completed=5 is a multiple of 5 but writer.py doesn't exist
        maybe_trigger_writer(tmp_path, n_completed=5, trigger_every=5)
        # No error; no file created

    def test_trigger_every_zero_no_op(self, tmp_path):
        maybe_trigger_writer(tmp_path, n_completed=10, trigger_every=0)

    def test_triggers_when_writer_present(self, tmp_path, monkeypatch):
        # Create a dummy writer.py that writes a sentinel file
        writer_path = Path('src') / 'writer.py'
        sentinel    = tmp_path / 'writer_triggered'
        # Monkeypatch subprocess.run to record the call
        calls = []
        monkeypatch.setattr('src.orchestrator.subprocess.run',
                            lambda *a, **k: calls.append(a))
        # Temporarily create the writer file
        writer_path.parent.mkdir(exist_ok=True)
        writer_path_existed = writer_path.exists()
        if not writer_path_existed:
            writer_path.write_text('# dummy')
        try:
            maybe_trigger_writer(tmp_path, n_completed=5, trigger_every=5)
            assert len(calls) == 1
        finally:
            if not writer_path_existed:
                writer_path.unlink(missing_ok=True)


# ============================================================================
# orchestrate — integration with mock train_fn
# ============================================================================

class TestOrchestrate:

    _YAML = """
        defaults:
          lr: 0.001
          batch_size: 8
          image_size: 8
          n_checkpoints: 2
          max_flat_sample: 50
          num_workers: 0
          writer_trigger_every: 2
        experiments:
          - dataset: PathMNIST
            epochs: 1
            seeds: [1, 2, 3]
    """

    def _run(self, tmp_path, yaml_override=None, resume=True,
             dry_run=False, train_fn=None):
        yaml_path = _write_yaml(tmp_path, yaml_override or self._YAML)
        results   = tmp_path / 'results'
        return orchestrate(
            config_path=yaml_path,
            results_dir=results,
            resume=resume,
            dry_run=dry_run,
            _train_fn=train_fn or _mock_train_fn,
        )

    def test_n_total_matches_grid(self, tmp_path):
        summary = self._run(tmp_path)
        assert summary['n_total'] == 3

    def test_all_runs_completed(self, tmp_path):
        summary = self._run(tmp_path)
        assert summary['n_completed'] == 3
        assert summary['n_failed']    == 0

    def test_status_json_created(self, tmp_path):
        self._run(tmp_path)
        assert (tmp_path / 'results' / 'status.json').exists()

    def test_all_runs_marked_completed(self, tmp_path):
        self._run(tmp_path)
        status = load_status(tmp_path / 'results')
        assert all(v == 'completed' for v in status.values())

    def test_results_csv_has_correct_row_count(self, tmp_path):
        self._run(tmp_path)
        with open(tmp_path / 'results' / 'results.csv') as f:
            assert len(list(csv.DictReader(f))) == 3

    def test_run_dirs_created(self, tmp_path):
        self._run(tmp_path)
        runs_dir = tmp_path / 'results' / 'runs'
        run_dirs = list(runs_dir.iterdir())
        assert len(run_dirs) == 3

    def test_second_call_skips_completed(self, tmp_path):
        self._run(tmp_path)
        summary2 = self._run(tmp_path)
        assert summary2['n_skipped']   == 3
        assert summary2['n_completed'] == 0

    def test_failed_run_retried(self, tmp_path):
        yaml_path = _write_yaml(tmp_path, self._YAML)
        results   = tmp_path / 'results'
        results.mkdir(parents=True, exist_ok=True)

        # Pre-mark one run as failed
        set_run_status('PathMNIST_1ep_seed1', 'failed', results)

        summary = orchestrate(
            config_path=yaml_path, results_dir=results,
            resume=True, _train_fn=_mock_train_fn,
        )
        assert get_run_status('PathMNIST_1ep_seed1', results) == 'completed'

    def test_failing_run_marked_failed_and_others_continue(self, tmp_path):
        def _failing_fn(config, output_dir, resume=True, _loaders=None):
            if config['seed'] == 2:
                raise RuntimeError("GPU OOM")
            return _mock_train_fn(config, output_dir, resume)

        summary = self._run(tmp_path, train_fn=_failing_fn)
        assert summary['n_failed']    == 1
        assert summary['n_completed'] == 2

        results = tmp_path / 'results'
        assert get_run_status('PathMNIST_1ep_seed2', results) == 'failed'

    def test_dry_run_prints_without_executing(self, tmp_path, capsys):
        summary = self._run(tmp_path, dry_run=True)
        out = capsys.readouterr().out
        assert 'PathMNIST' in out
        # No runs directory created (nothing actually ran)
        runs_dir = tmp_path / 'results' / 'runs'
        assert not any(runs_dir.iterdir()) if runs_dir.exists() else True

    def test_writer_triggered_at_correct_count(self, tmp_path, monkeypatch):
        calls = []
        monkeypatch.setattr('src.orchestrator.subprocess.run',
                            lambda *a, **k: calls.append(a))
        # Create a dummy writer.py so the trigger doesn't short-circuit
        writer_path = Path('src') / 'writer.py'
        existed = writer_path.exists()
        if not existed:
            writer_path.write_text('# dummy')
        try:
            self._run(tmp_path)  # 3 runs, trigger_every=2 → fires at 2 completed
            assert len(calls) == 1
        finally:
            if not existed:
                writer_path.unlink(missing_ok=True)

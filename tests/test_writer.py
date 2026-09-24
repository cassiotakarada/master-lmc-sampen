"""
tests/test_writer.py — Unit tests for src/writer.py.

All tests use tmp_path for file I/O.  Claude CLI is mocked via monkeypatch
so no network calls are made.

Covers:
  - Lock management: lock/unlock/is_locked/persistence
  - Provenance: load/update round-trip
  - wrap_claudedraft: correct LaTeX output
  - write_tex_section: insert into file, replace existing, atomic write
  - read_tex_section: reads current content between markers
  - summarise_results: handles missing CSV gracefully
  - build_prompt: includes section instruction and data
  - write_section: full pipeline with mocked call_claude
  - print_status: runs without crash
  - main CLI: --status, --lock, --unlock, --dry-run, --section
"""

import json
import csv
from pathlib import Path

import pytest

import src.writer as writer
from src.writer import (
    _BEGIN_FMT,
    _END_FMT,
    build_prompt,
    is_locked,
    load_locks,
    load_provenance,
    lock_section,
    print_status,
    read_tex_section,
    summarise_results,
    unlock_section,
    update_provenance,
    wrap_claudedraft,
    write_section,
    write_tex_section,
    main,
)


# ============================================================================
# Shared helpers
# ============================================================================

def _locks(tmp_path: Path) -> Path:
    return tmp_path / '.writer_locks.json'


def _prov(tmp_path: Path) -> Path:
    return tmp_path / '.writer_provenance.json'


def _tex(tmp_path: Path, name: str = 'results.tex') -> Path:
    p = tmp_path / name
    p.write_text('% preamble\n', encoding='utf-8')
    return p


def _write_results_csv(results_dir: Path, n_runs: int = 3) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / 'results.csv'
    rows = [
        {
            'run_id':         f'PathMNIST_30ep_seed{s}',
            'dataset':        'PathMNIST',
            'epochs':         30,
            'seed':           s,
            'final_val_acc':  0.90 + s * 0.01,
            'final_val_loss': 0.25,
            'completed_at':   '2025-01-01T00:00:00+00:00',
        }
        for s in range(n_runs)
    ]
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)


# ============================================================================
# Lock management
# ============================================================================

class TestLocks:

    def test_not_locked_by_default(self, tmp_path):
        assert not is_locked('results/overview', _locks(tmp_path))

    def test_lock_and_is_locked(self, tmp_path):
        lock_section('results/overview', locks_file=_locks(tmp_path))
        assert is_locked('results/overview', _locks(tmp_path))

    def test_unlock_removes_lock(self, tmp_path):
        lock_section('results/overview', locks_file=_locks(tmp_path))
        unlock_section('results/overview', locks_file=_locks(tmp_path))
        assert not is_locked('results/overview', _locks(tmp_path))

    def test_unlock_nonexistent_does_not_raise(self, tmp_path):
        unlock_section('nonexistent', locks_file=_locks(tmp_path))  # no exception

    def test_lock_persisted_to_json(self, tmp_path):
        lock_section('results/overview', locks_file=_locks(tmp_path))
        data = json.loads(_locks(tmp_path).read_text())
        assert 'results/overview' in data

    def test_lock_includes_timestamp(self, tmp_path):
        lock_section('results/overview', locks_file=_locks(tmp_path))
        data = json.loads(_locks(tmp_path).read_text())
        assert 'locked_at' in data['results/overview']

    def test_multiple_sections_tracked_independently(self, tmp_path):
        lock_section('results/overview',     locks_file=_locks(tmp_path))
        lock_section('results/lmc_analysis', locks_file=_locks(tmp_path))
        assert is_locked('results/overview',     _locks(tmp_path))
        assert is_locked('results/lmc_analysis', _locks(tmp_path))

    def test_load_locks_empty_when_no_file(self, tmp_path):
        assert load_locks(_locks(tmp_path)) == {}


# ============================================================================
# Provenance
# ============================================================================

class TestProvenance:

    def test_load_empty_when_no_file(self, tmp_path):
        assert load_provenance(_prov(tmp_path)) == {}

    def test_update_and_load_round_trip(self, tmp_path):
        update_provenance('results/overview',
                          {'model': 'claude-opus-4-7', 'n_runs': 5},
                          _prov(tmp_path))
        prov = load_provenance(_prov(tmp_path))
        assert prov['results/overview']['model'] == 'claude-opus-4-7'
        assert prov['results/overview']['n_runs'] == 5

    def test_generated_at_added_automatically(self, tmp_path):
        update_provenance('results/overview', {}, _prov(tmp_path))
        prov = load_provenance(_prov(tmp_path))
        assert 'generated_at' in prov['results/overview']

    def test_update_overwrites_previous(self, tmp_path):
        update_provenance('results/overview', {'n_runs': 1}, _prov(tmp_path))
        update_provenance('results/overview', {'n_runs': 9}, _prov(tmp_path))
        prov = load_provenance(_prov(tmp_path))
        assert prov['results/overview']['n_runs'] == 9


# ============================================================================
# wrap_claudedraft
# ============================================================================

class TestWrapClaudedraft:

    def test_begins_with_env(self):
        out = wrap_claudedraft('hello')
        assert out.startswith('\\begin{claudedraft}')

    def test_ends_with_env(self):
        out = wrap_claudedraft('hello')
        assert out.endswith('\\end{claudedraft}')

    def test_content_included(self):
        out = wrap_claudedraft('my content here')
        assert 'my content here' in out

    def test_strips_leading_trailing_whitespace(self):
        out = wrap_claudedraft('  trimmed  ')
        assert 'trimmed' in out
        assert '  trimmed  ' not in out


# ============================================================================
# write_tex_section / read_tex_section
# ============================================================================

class TestTexSection:

    def test_appends_when_no_marker(self, tmp_path):
        tex = _tex(tmp_path)
        write_tex_section(tex, 'results/overview', 'NEW CONTENT')
        text = tex.read_text()
        assert _BEGIN_FMT.format('results/overview') in text
        assert 'NEW CONTENT' in text

    def test_replaces_existing_content(self, tmp_path):
        tex = _tex(tmp_path)
        write_tex_section(tex, 'results/overview', 'FIRST')
        write_tex_section(tex, 'results/overview', 'SECOND')
        text = tex.read_text()
        assert 'FIRST'  not in text
        assert 'SECOND' in text

    def test_returns_true_on_change(self, tmp_path):
        tex = _tex(tmp_path)
        assert write_tex_section(tex, 'results/overview', 'content') is True

    def test_returns_false_when_unchanged(self, tmp_path):
        tex = _tex(tmp_path)
        write_tex_section(tex, 'results/overview', 'content')
        assert write_tex_section(tex, 'results/overview', 'content') is False

    def test_creates_parent_dir_if_missing(self, tmp_path):
        tex = tmp_path / 'sub' / 'nested.tex'
        write_tex_section(tex, 'results/overview', 'content')
        assert tex.exists()

    def test_read_returns_none_when_missing_file(self, tmp_path):
        assert read_tex_section(tmp_path / 'missing.tex', 'x') is None

    def test_read_returns_none_when_no_marker(self, tmp_path):
        tex = _tex(tmp_path)
        assert read_tex_section(tex, 'nonexistent_section') is None

    def test_read_returns_current_content(self, tmp_path):
        tex = _tex(tmp_path)
        write_tex_section(tex, 'results/overview', 'EXPECTED')
        content = read_tex_section(tex, 'results/overview')
        assert content is not None
        assert 'EXPECTED' in content

    def test_multiple_sections_coexist(self, tmp_path):
        tex = _tex(tmp_path)
        write_tex_section(tex, 'results/overview',  'A')
        write_tex_section(tex, 'results/lmc_analysis', 'B')
        assert 'A' in tex.read_text()
        assert 'B' in tex.read_text()


# ============================================================================
# summarise_results
# ============================================================================

class TestSummariseResults:

    def test_returns_empty_when_no_csv(self, tmp_path):
        s = summarise_results(tmp_path)
        assert s['n_completed'] == 0

    def test_n_completed_matches_rows(self, tmp_path):
        _write_results_csv(tmp_path, n_runs=5)
        s = summarise_results(tmp_path)
        assert s['n_completed'] == 5

    def test_datasets_populated(self, tmp_path):
        _write_results_csv(tmp_path, n_runs=2)
        s = summarise_results(tmp_path)
        assert 'PathMNIST' in s['datasets']

    def test_mean_val_acc_computed(self, tmp_path):
        _write_results_csv(tmp_path, n_runs=3)
        s = summarise_results(tmp_path)
        assert s['mean_val_acc'] is not None
        assert 0.0 <= s['mean_val_acc'] <= 1.0


# ============================================================================
# build_prompt
# ============================================================================

class TestBuildPrompt:

    def test_contains_section_instruction(self):
        data  = {'n_completed': 3, 'runs': [], 'metrics': []}
        prompt = build_prompt('results/overview', data)
        assert 'overview' in prompt.lower() or 'experimental setup' in prompt.lower()

    def test_contains_data_json(self):
        data = {'n_completed': 7, 'runs': [], 'metrics': []}
        assert '7' in build_prompt('results/overview', data)

    def test_falls_back_for_unknown_section(self):
        data   = {'n_completed': 1, 'runs': [], 'metrics': []}
        prompt = build_prompt('unknown/section', data)
        assert len(prompt) > 50   # still returns something useful

    def test_truncates_large_data(self):
        data  = {'n_completed': 1, 'big': 'x' * 10_000}
        prompt = build_prompt('results/overview', data)
        assert len(prompt) < 12_000   # sanity cap


# ============================================================================
# write_section — mocked Claude
# ============================================================================

class TestWriteSection:

    def _patch(self, monkeypatch, return_value='Generated text.'):
        monkeypatch.setattr(writer, 'call_claude', lambda *a, **kw: return_value)

    def test_skips_locked_section(self, tmp_path, monkeypatch):
        self._patch(monkeypatch)
        _write_results_csv(tmp_path)
        lock_section('results/overview', locks_file=_locks(tmp_path))

        result = write_section(
            'results/overview', tmp_path,
            locks_file=_locks(tmp_path),
            prov_file=_prov(tmp_path),
        )
        assert result is False

    def test_skips_when_too_few_runs(self, tmp_path, monkeypatch):
        self._patch(monkeypatch)
        # lmc_analysis requires min_runs=3; provide only 1
        _write_results_csv(tmp_path, n_runs=1)
        result = write_section(
            'results/lmc_analysis', tmp_path,
            locks_file=_locks(tmp_path),
            prov_file=_prov(tmp_path),
        )
        assert result is False

    def test_writes_content_to_tex_file(self, tmp_path, monkeypatch):
        self._patch(monkeypatch, 'LaTeX content here.')
        _write_results_csv(tmp_path)

        # Override SECTIONS to use tmp_path for the tex file
        tex = tmp_path / 'resultados.tex'
        tex.write_text('% preamble\n')
        monkeypatch.setattr(
            writer, 'SECTIONS',
            {'results/overview': {'file': str(tex), 'min_runs': 1}},
        )

        write_section(
            'results/overview', tmp_path,
            locks_file=_locks(tmp_path),
            prov_file=_prov(tmp_path),
        )
        assert 'LaTeX content here.' in tex.read_text()

    def test_content_wrapped_in_claudedraft(self, tmp_path, monkeypatch):
        self._patch(monkeypatch, 'My paragraph.')
        _write_results_csv(tmp_path)

        tex = tmp_path / 'resultados.tex'
        tex.write_text('% preamble\n')
        monkeypatch.setattr(
            writer, 'SECTIONS',
            {'results/overview': {'file': str(tex), 'min_runs': 1}},
        )

        write_section(
            'results/overview', tmp_path,
            locks_file=_locks(tmp_path),
            prov_file=_prov(tmp_path),
        )
        text = tex.read_text()
        assert '\\begin{claudedraft}' in text
        assert '\\end{claudedraft}'   in text

    def test_provenance_updated_after_write(self, tmp_path, monkeypatch):
        self._patch(monkeypatch)
        _write_results_csv(tmp_path)

        tex = tmp_path / 'resultados.tex'
        tex.write_text('% preamble\n')
        monkeypatch.setattr(
            writer, 'SECTIONS',
            {'results/overview': {'file': str(tex), 'min_runs': 1}},
        )

        write_section(
            'results/overview', tmp_path,
            locks_file=_locks(tmp_path),
            prov_file=_prov(tmp_path),
        )
        prov = load_provenance(_prov(tmp_path))
        assert 'results/overview' in prov

    def test_returns_false_when_claude_unavailable(self, tmp_path, monkeypatch):
        monkeypatch.setattr(writer, 'call_claude', lambda *a, **kw: None)
        _write_results_csv(tmp_path)

        tex = tmp_path / 'resultados.tex'
        tex.write_text('% preamble\n')
        monkeypatch.setattr(
            writer, 'SECTIONS',
            {'results/overview': {'file': str(tex), 'min_runs': 1}},
        )

        result = write_section(
            'results/overview', tmp_path,
            locks_file=_locks(tmp_path),
            prov_file=_prov(tmp_path),
        )
        assert result is False

    def test_dry_run_does_not_write_file(self, tmp_path, monkeypatch):
        called = []
        monkeypatch.setattr(writer, 'call_claude',
                            lambda *a, **kw: called.append(1) or 'text')
        _write_results_csv(tmp_path)

        tex = tmp_path / 'resultados.tex'
        tex.write_text('% preamble\n')
        original = tex.read_text()
        monkeypatch.setattr(
            writer, 'SECTIONS',
            {'results/overview': {'file': str(tex), 'min_runs': 1}},
        )

        result = write_section(
            'results/overview', tmp_path, dry_run=True,
            locks_file=_locks(tmp_path),
            prov_file=_prov(tmp_path),
        )
        assert result is True
        assert tex.read_text() == original   # file not touched
        assert len(called) == 0             # claude not called


# ============================================================================
# print_status
# ============================================================================

class TestPrintStatus:

    def test_runs_without_crash(self, tmp_path, capsys):
        print_status(tmp_path, _locks(tmp_path), _prov(tmp_path))
        out = capsys.readouterr().out
        assert 'results/overview' in out

    def test_shows_locked_label(self, tmp_path, capsys):
        lock_section('results/overview', locks_file=_locks(tmp_path))
        print_status(tmp_path, _locks(tmp_path), _prov(tmp_path))
        out = capsys.readouterr().out
        assert 'LOCKED' in out


# ============================================================================
# main() CLI
# ============================================================================

class TestMain:

    def _patch_sections(self, monkeypatch, tmp_path):
        tex = tmp_path / 'resultados.tex'
        tex.write_text('% preamble\n')
        monkeypatch.setattr(
            writer, 'SECTIONS',
            {'results/overview': {'file': str(tex), 'min_runs': 1}},
        )
        monkeypatch.setattr(writer, 'LOCKS_FILE',    _locks(tmp_path))
        monkeypatch.setattr(writer, 'PROVENANCE_FILE', _prov(tmp_path))
        return tex

    def test_status_does_not_raise(self, tmp_path, capsys, monkeypatch):
        self._patch_sections(monkeypatch, tmp_path)
        main(['--status', '--results-dir', str(tmp_path)])

    def test_lock_via_cli(self, tmp_path, monkeypatch):
        self._patch_sections(monkeypatch, tmp_path)
        monkeypatch.setattr(writer, 'TESE_DIR', tmp_path)
        main(['--lock', 'results/overview', '--results-dir', str(tmp_path)])
        assert is_locked('results/overview', _locks(tmp_path))

    def test_unlock_via_cli(self, tmp_path, monkeypatch):
        self._patch_sections(monkeypatch, tmp_path)
        monkeypatch.setattr(writer, 'TESE_DIR', tmp_path)
        lock_section('results/overview', locks_file=_locks(tmp_path))
        main(['--unlock', 'results/overview', '--results-dir', str(tmp_path)])
        assert not is_locked('results/overview', _locks(tmp_path))

    def test_dry_run_does_not_call_claude(self, tmp_path, monkeypatch):
        self._patch_sections(monkeypatch, tmp_path)
        _write_results_csv(tmp_path)
        called = []
        monkeypatch.setattr(writer, 'call_claude',
                            lambda *a, **kw: called.append(1))
        main(['--dry-run', '--results-dir', str(tmp_path)])
        assert len(called) == 0

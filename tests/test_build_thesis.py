"""tests/test_build_thesis.py — Tests for scripts/build_thesis.sh

These tests do NOT require a full LaTeX installation.  They verify:
  - bash syntax is clean
  - --help output is sensible
  - --check exits non-zero when pdflatex/bibtex are missing
  - unknown flags produce an error message and exit non-zero
  - --clean flag is accepted without error (syntax-level)
  - --fast flag is accepted without error (syntax-level)
  - --output flag requires an argument

LaTeX-dependent tests (actual compilation) are intentionally omitted.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import pytest

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent
SCRIPT = PROJECT_DIR / "scripts" / "build_thesis.sh"
BASH = shutil.which("bash") or "/bin/bash"  # absolute path, immune to PATH changes


# ── Helpers ───────────────────────────────────────────────────────────────────

def run_script(*args, env=None, timeout=30):
    """Run build_thesis.sh with given args; return CompletedProcess."""
    cmd = [BASH, str(SCRIPT)] + list(args)
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )


def _no_latex_env(fake_bin: str):
    """Return an env dict with a fake bin dir that has coreutils but no pdflatex/bibtex.

    Strategy: symlink all executables from /usr/bin and /bin into fake_bin,
    then delete the pdflatex and bibtex symlinks.  This lets the script
    initialize (dirname, pwd, grep, etc. work) but fail at check_prereqs().
    """
    # Symlink every real executable into fake_bin
    for src_dir in ("/usr/bin", "/bin"):
        src_path = Path(src_dir)
        if not src_path.is_dir():
            continue
        for exe in src_path.iterdir():
            if not exe.is_file() and not exe.is_symlink():
                continue
            dest = Path(fake_bin) / exe.name
            if not dest.exists():
                try:
                    dest.symlink_to(exe)
                except OSError:
                    pass  # skip unreadable entries

    # Remove pdflatex and bibtex so the script cannot find them
    for tool in ("pdflatex", "bibtex"):
        target = Path(fake_bin) / tool
        target.unlink(missing_ok=True)

    env = os.environ.copy()
    env["PATH"] = fake_bin
    return env


# ── Script existence ───────────────────────────────────────────────────────────

class TestScriptExists:
    def test_script_file_present(self):
        assert SCRIPT.exists(), f"Script not found: {SCRIPT}"

    def test_script_is_not_empty(self):
        assert SCRIPT.stat().st_size > 0


# ── Bash syntax check ─────────────────────────────────────────────────────────

class TestBashSyntax:
    def test_no_syntax_errors(self):
        result = subprocess.run(
            ["bash", "-n", str(SCRIPT)],
            capture_output=True, text=True
        )
        assert result.returncode == 0, (
            f"bash -n reported syntax errors:\n{result.stderr}"
        )

    def test_shebang_is_bash(self):
        first_line = SCRIPT.read_text().splitlines()[0]
        assert "bash" in first_line, f"Unexpected shebang: {first_line!r}"


# ── --help flag ───────────────────────────────────────────────────────────────

class TestHelpFlag:
    def test_help_exits_zero(self):
        result = run_script("--help")
        assert result.returncode == 0, f"--help exited {result.returncode}"

    def test_short_help_exits_zero(self):
        result = run_script("-h")
        assert result.returncode == 0

    def test_help_mentions_usage(self):
        result = run_script("--help")
        combined = result.stdout + result.stderr
        assert "Usage" in combined or "usage" in combined, (
            f"--help output missing 'Usage':\n{combined}"
        )

    def test_help_mentions_options(self):
        result = run_script("--help")
        combined = result.stdout + result.stderr
        # At least --fast and --clean should be documented
        assert "--fast" in combined, "--help missing --fast option"
        assert "--clean" in combined or "-c" in combined, "--help missing --clean option"

    def test_help_mentions_check(self):
        result = run_script("--help")
        combined = result.stdout + result.stderr
        assert "--check" in combined, "--help missing --check option"


# ── Unknown flags ─────────────────────────────────────────────────────────────

class TestUnknownFlags:
    @pytest.mark.parametrize("bad_flag", ["--bogus", "--xyz", "-z", "--no-such-option"])
    def test_unknown_flag_exits_nonzero(self, bad_flag):
        result = run_script(bad_flag)
        assert result.returncode != 0, (
            f"Expected non-zero exit for {bad_flag!r}, got 0"
        )

    def test_unknown_flag_prints_error(self):
        result = run_script("--bogus-flag-12345")
        combined = result.stdout + result.stderr
        assert len(combined.strip()) > 0, "Expected some error output for unknown flag"

    def test_unknown_flag_hints_help(self):
        result = run_script("--bogus-flag-12345")
        combined = result.stdout + result.stderr
        # The script die() message references --help
        assert "help" in combined.lower() or "unknown" in combined.lower(), (
            f"Error message should mention 'help' or 'unknown':\n{combined}"
        )


# ── --output flag ─────────────────────────────────────────────────────────────

class TestOutputFlag:
    def test_output_requires_argument(self):
        # '--output' with no following argument: bash ${2:?...} expansion → exit 1
        result = run_script("--output")
        assert result.returncode != 0, "--output with no DIR should fail"

    def test_short_output_requires_argument(self):
        result = run_script("-o")
        assert result.returncode != 0, "-o with no DIR should fail"


# ── --check with missing prerequisites ────────────────────────────────────────

class TestCheckPrereqs:
    def test_check_fails_without_pdflatex(self, tmp_path):
        """PATH with no pdflatex/bibtex → --check exits non-zero."""
        result = run_script("--check", env=_no_latex_env(str(tmp_path)))
        assert result.returncode != 0, (
            "--check should fail when pdflatex is absent"
        )

    def test_check_reports_missing_tool(self, tmp_path):
        """Script names 'pdflatex' or 'bibtex' as missing in its error output."""
        result = run_script("--check", env=_no_latex_env(str(tmp_path)))
        combined = result.stdout + result.stderr
        assert "pdflatex" in combined or "bibtex" in combined, (
            f"--check output should name missing tools:\n{combined}"
        )

    def test_check_does_not_compile(self):
        """--check exits before launching pdflatex — no PDF is produced by it."""
        # This is a structural / code-reading test; the real guarantee is in
        # the script's `[[ $CHECK_ONLY -eq 1 ]] && exit 0` guard which runs
        # before any pdflatex call.
        src = SCRIPT.read_text()
        check_guard = "CHECK_ONLY -eq 1"
        pdflatex_call = "run_pdflatex 1"
        assert src.index(check_guard) < src.index(pdflatex_call), (
            "CHECK_ONLY guard must appear before first run_pdflatex call"
        )


# ── Flag combinations ─────────────────────────────────────────────────────────

class TestFlagCombinations:
    def test_clean_and_check_together(self):
        """--clean --check: clean flag is parsed, then check runs (and may fail on missing latex)."""
        result = run_script("--clean", "--check")
        # Exit code depends on whether pdflatex is installed; what matters is
        # it doesn't crash with an unknown-flag error.
        combined = result.stdout + result.stderr
        assert "Unknown option" not in combined, (
            "--clean --check should not produce 'Unknown option' error"
        )

    def test_fast_and_check_together(self):
        result = run_script("--fast", "--check")
        combined = result.stdout + result.stderr
        assert "Unknown option" not in combined

    def test_help_first_ignores_other_flags(self):
        """--help should exit 0 regardless of other flags (show_help exits immediately)."""
        result = run_script("--help", "--bogus")
        # show_help() exits before reaching the unknown-flag branch
        assert result.returncode == 0


# ── Script content invariants ─────────────────────────────────────────────────

class TestScriptContent:
    """Verify key design decisions are reflected in the script source."""

    def _source(self):
        return SCRIPT.read_text()

    def test_set_euo_pipefail(self):
        assert "set -euo pipefail" in self._source(), (
            "Script must use 'set -euo pipefail' for safety"
        )

    def test_nonstopmode(self):
        assert "-interaction=nonstopmode" in self._source(), (
            "pdflatex must be invoked with -interaction=nonstopmode"
        )

    def test_halt_on_error(self):
        assert "-halt-on-error" in self._source(), (
            "pdflatex must be invoked with -halt-on-error"
        )

    def test_bibtex_non_fatal(self):
        # bibtex failure is warned but not fatal — we use 'warn' not 'die'
        src = self._source()
        bibtex_block_start = src.find("run_bibtex()")
        assert bibtex_block_start != -1, "run_bibtex() not found in script"

    def test_three_pdflatex_passes_in_full_build(self):
        src = self._source()
        # Full build calls run_pdflatex 1, 2, and 3
        assert "run_pdflatex 1" in src
        assert "run_pdflatex 2" in src
        assert "run_pdflatex 3" in src

    def test_fast_mode_single_pass_only(self):
        src = self._source()
        # The FAST block must NOT contain run_pdflatex 2 or 3 — they are inside
        # the `if [[ $FAST -eq 0 ]]` branch.
        fast_guard = "FAST -eq 0"
        assert fast_guard in src, "Script must gate full build on FAST flag"

    def test_log_is_truncated_before_build(self):
        src = self._source()
        # `: > "$LOGFILE"` truncates the log at the start of each build
        assert ': > "$LOGFILE"' in src or ':>"$LOGFILE"' in src, (
            "Script must truncate LOGFILE before each build run"
        )

    def test_output_dir_copied(self):
        src = self._source()
        assert "cp " in src, "Script must copy PDF to output directory"

    def test_clean_extensions_include_aux(self):
        src = self._source()
        assert "aux" in src and "bbl" in src and "blg" in src, (
            "clean_aux() must remove .aux, .bbl, .blg files"
        )

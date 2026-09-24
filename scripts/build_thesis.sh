#!/usr/bin/env bash
# scripts/build_thesis.sh — Compile the thesis PDF on the Titan machine.
#
# Full build:  pdflatex → bibtex → pdflatex → pdflatex
#
# Usage:
#   bash scripts/build_thesis.sh [OPTIONS]
#
# Options:
#   -c | --clean        Remove auxiliary files before building
#   -f | --fast         Single pdflatex pass — skip bibliography (draft mode)
#   -o | --output DIR   Copy the final PDF here  (default: project root)
#   -h | --help         Show this message and exit
#   --check             Verify prerequisites only — do not compile
#
# Prerequisites:
#   pdflatex  (texlive-latex-base or texlive-full)
#   bibtex    (texlive-bibtex-extra)
#   For claudedraft.sty: mdframed, xcolor  (texlive-latex-extra)
#
#   Install on Debian/Ubuntu (requires sudo on first run):
#     sudo apt install texlive-full
#
# Exit codes:
#   0  PDF compiled successfully (or prerequisites OK for --check)
#   1  Missing prerequisite or compilation error

set -euo pipefail

# ─── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
TESE_DIR="$PROJECT_DIR/tese"
MAIN="main"
LOGFILE="$TESE_DIR/build.log"

# ─── Defaults ─────────────────────────────────────────────────────────────────
CLEAN=0
FAST=0
CHECK_ONLY=0
OUTPUT_DIR="$PROJECT_DIR"

# ─── Colours ──────────────────────────────────────────────────────────────────
if [[ -t 1 ]]; then
  RED='\033[0;31m'; GREEN='\033[0;32m'
  YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
else
  RED=''; GREEN=''; YELLOW=''; CYAN=''; NC=''
fi

info()    { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; }
step()    { echo -e "${CYAN}[STEP]${NC}  $*"; }
die()     { error "$*"; exit 1; }

# ─── Argument parsing ─────────────────────────────────────────────────────────
show_help() {
  grep '^#' "$0" | grep -v '#!/' | sed 's/^# \{0,1\}//'
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)       show_help ;;
    -c|--clean)      CLEAN=1; shift ;;
    -f|--fast)       FAST=1; shift ;;
    --check)         CHECK_ONLY=1; shift ;;
    -o|--output)     OUTPUT_DIR="${2:?--output requires a directory}"; shift 2 ;;
    *) die "Unknown option: $1  (use --help for usage)" ;;
  esac
done

# ─── Prerequisite check ───────────────────────────────────────────────────────
check_prereqs() {
  local ok=1

  for cmd in pdflatex bibtex; do
    if command -v "$cmd" &>/dev/null; then
      info "Found: $cmd  ($(command -v "$cmd"))"
    else
      error "Not found: $cmd"
      error "  Install: sudo apt install texlive-full"
      ok=0
    fi
  done

  if [[ ! -f "$TESE_DIR/$MAIN.tex" ]]; then
    error "Main LaTeX file not found: $TESE_DIR/$MAIN.tex"
    ok=0
  fi

  if [[ ! -f "$TESE_DIR/claudedraft.sty" ]]; then
    warn "claudedraft.sty not found — orange draft boxes will not render"
  fi

  if [[ $ok -eq 0 ]]; then
    return 1
  fi
  info "All prerequisites satisfied"
  return 0
}

# ─── Auxiliary file cleanup ───────────────────────────────────────────────────
clean_aux() {
  step "Cleaning auxiliary files..."
  local exts=(aux log bbl blg toc lof lot out fls fdb_latexmk nav snm vrb)
  for ext in "${exts[@]}"; do
    find "$TESE_DIR" -maxdepth 2 -name "*.$ext" -print -delete 2>/dev/null || true
  done
  # Remove old PDF so we can detect a fresh build failure
  rm -f "$TESE_DIR/$MAIN.pdf"
  info "Clean done"
}

# ─── pdflatex invocation ──────────────────────────────────────────────────────
run_pdflatex() {
  local pass="$1"
  step "pdflatex pass $pass..."
  if ! pdflatex \
      -interaction=nonstopmode \
      -halt-on-error \
      -output-directory="$TESE_DIR" \
      "$TESE_DIR/$MAIN.tex" >> "$LOGFILE" 2>&1; then
    error "pdflatex pass $pass failed — see $LOGFILE"
    # Print the last 30 relevant lines to help diagnose
    echo "── Last lines of $LOGFILE ──────────────────────"
    grep -E '(^!|Error|error|Warning.*fatal)' "$LOGFILE" | tail -20 || true
    echo "────────────────────────────────────────────────"
    return 1
  fi
}

# ─── bibtex invocation ────────────────────────────────────────────────────────
run_bibtex() {
  step "bibtex..."
  # bibtex must be run from the directory containing the .aux file
  if ! (cd "$TESE_DIR" && bibtex "$MAIN" >> "$LOGFILE" 2>&1); then
    warn "bibtex returned non-zero — references may be incomplete"
    warn "Check $LOGFILE for details"
    # Non-fatal: continue with remaining pdflatex passes
  fi
}

# ─── Post-build report ────────────────────────────────────────────────────────
report_pdf() {
  local pdf="$TESE_DIR/$MAIN.pdf"
  if [[ ! -f "$pdf" ]]; then
    die "Build finished but PDF not found at $pdf"
  fi

  local size
  size=$(du -sh "$pdf" | cut -f1)

  # Count pages via pdfinfo if available
  local pages="unknown"
  if command -v pdfinfo &>/dev/null; then
    pages=$(pdfinfo "$pdf" 2>/dev/null | awk '/^Pages:/ {print $2}') || true
  fi

  info "PDF:   $pdf"
  info "Size:  $size   Pages: $pages"
}

# ─── Main ─────────────────────────────────────────────────────────────────────
main() {
  echo ""
  info "Thesis build script — project: $PROJECT_DIR"
  info "Output directory:  $OUTPUT_DIR"
  info "Fast mode: $( [[ $FAST -eq 1 ]] && echo yes || echo no )"
  echo ""

  check_prereqs || die "Prerequisite check failed"
  [[ $CHECK_ONLY -eq 1 ]] && exit 0

  mkdir -p "$OUTPUT_DIR"
  : > "$LOGFILE"   # truncate / create log

  [[ $CLEAN -eq 1 ]] && clean_aux

  # Pass 1: generate .aux / .toc
  run_pdflatex 1

  if [[ $FAST -eq 0 ]]; then
    # Bibliography
    run_bibtex

    # Pass 2: resolve citations
    run_pdflatex 2

    # Pass 3: fix cross-references / TOC
    run_pdflatex 3
  fi

  report_pdf

  # Copy to output directory if different from tese/
  local pdf="$TESE_DIR/$MAIN.pdf"
  local dest="$OUTPUT_DIR/$MAIN.pdf"
  if [[ "$(realpath "$pdf")" != "$(realpath "$dest" 2>/dev/null || echo __none__)" ]]; then
    cp "$pdf" "$dest"
    info "Copied → $dest"
  fi

  echo ""
  info "Build complete ✓"
  echo ""
}

main "$@"

#!/usr/bin/env bash
# Polls the orchestrator running Phase-1 (subsampling) grid and runs the
# figure script when all 9 runs complete (or the orchestrator process dies).
#
# Output:
#   results/_watch_overfit.log   — heartbeat + final status
#   results/overfit_figs/*.{pdf,png,csv,tex}  — figures + tables

set -euo pipefail
cd "$(dirname "$0")/.."   # repo root

LOG="results/_watch_overfit.log"
STATUS="results/status.json"
TARGET_KEYS='PathMNIST_100ep_seed42_frac0.05 PathMNIST_100ep_seed42_frac0.10 PathMNIST_100ep_seed42_frac0.25 PathMNIST_100ep_seed123_frac0.05 PathMNIST_100ep_seed123_frac0.10 PathMNIST_100ep_seed123_frac0.25 PathMNIST_100ep_seed2024_frac0.05 PathMNIST_100ep_seed2024_frac0.10 PathMNIST_100ep_seed2024_frac0.25'

echo "[$(date -Iseconds)] watcher started, polling every 60s" > "$LOG"

while true; do
    if [ ! -f "$STATUS" ]; then
        echo "[$(date -Iseconds)] no status.json yet" >> "$LOG"
        sleep 60; continue
    fi

    # Count completed
    completed=$(python3 - <<PY
import json
st = json.load(open("$STATUS"))
keys = "$TARGET_KEYS".split()
done = sum(1 for k in keys if st.get(k) == "completed")
failed = sum(1 for k in keys if st.get(k) == "failed")
print(f"{done} {failed}", end="")
PY
    )
    done_n=$(echo "$completed" | awk '{print $1}')
    failed_n=$(echo "$completed" | awk '{print $2}')

    # Orchestrator still alive?
    if pgrep -af "src.orchestrator.*experiment_grid_overfit" > /dev/null; then
        alive=yes
    else
        alive=no
    fi

    echo "[$(date -Iseconds)] completed=$done_n/9 failed=$failed_n orchestrator_alive=$alive" >> "$LOG"

    if [ "$done_n" -eq 9 ]; then
        echo "[$(date -Iseconds)] all 9 runs completed — triggering figures" >> "$LOG"
        ./.venv/bin/python scripts/make_overfit_figs.py >> "$LOG" 2>&1
        echo "[$(date -Iseconds)] figures done; exiting watcher" >> "$LOG"
        exit 0
    fi

    if [ "$alive" = "no" ] && [ "$done_n" -lt 9 ]; then
        echo "[$(date -Iseconds)] orchestrator gone with done=$done_n/9 failed=$failed_n — running figs on partial results" >> "$LOG"
        ./.venv/bin/python scripts/make_overfit_figs.py >> "$LOG" 2>&1 || true
        echo "[$(date -Iseconds)] partial figures done; exiting watcher" >> "$LOG"
        exit 0
    fi

    sleep 60
done

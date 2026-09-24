#!/usr/bin/env bash
# Polls the orchestrator running Phase-3 (architecture sweep) and runs
# the figure script when all 6 runs complete (or the orchestrator dies).
set -euo pipefail
cd "$(dirname "$0")/.."

LOG="results/_watch_arch.log"
STATUS="results/status.json"
TARGET_KEYS='PathMNIST_100ep_seed42_arch-resnet34 PathMNIST_100ep_seed42_arch-densenet121 PathMNIST_100ep_seed123_arch-resnet34 PathMNIST_100ep_seed123_arch-densenet121 PathMNIST_100ep_seed2024_arch-resnet34 PathMNIST_100ep_seed2024_arch-densenet121'

echo "[$(date -Iseconds)] ARCH watcher started, polling every 60s" > "$LOG"

while true; do
    if [ ! -f "$STATUS" ]; then
        echo "[$(date -Iseconds)] no status.json yet" >> "$LOG"
        sleep 60; continue
    fi

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

    if pgrep -af "src.orchestrator.*experiment_grid_arch" > /dev/null; then
        alive=yes
    else
        alive=no
    fi

    echo "[$(date -Iseconds)] completed=$done_n/6 failed=$failed_n orchestrator_alive=$alive" >> "$LOG"

    if [ "$done_n" -eq 6 ]; then
        echo "[$(date -Iseconds)] all 6 runs completed — triggering figures" >> "$LOG"
        ./.venv/bin/python scripts/make_arch_figs.py >> "$LOG" 2>&1
        echo "[$(date -Iseconds)] ARCH figures done; exiting watcher" >> "$LOG"
        exit 0
    fi

    if [ "$alive" = "no" ] && [ "$done_n" -lt 6 ]; then
        echo "[$(date -Iseconds)] orchestrator gone with done=$done_n/6 — running figs on partial results" >> "$LOG"
        ./.venv/bin/python scripts/make_arch_figs.py >> "$LOG" 2>&1 || true
        echo "[$(date -Iseconds)] partial figures done; exiting watcher" >> "$LOG"
        exit 0
    fi

    sleep 60
done

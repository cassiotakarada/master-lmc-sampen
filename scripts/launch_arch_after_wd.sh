#!/usr/bin/env bash
# Chained launcher:
#   1. Poll until Phase-2 (weight decay) sweep finishes — all 9 runs completed
#      AND the orchestrator process has exited (GPU genuinely free).
#   2. Then launch Phase-3 (architecture) orchestrator + watcher in background.
#   3. Exit.
#
# Output:
#   results/_launch_arch.log  — heartbeat + final status
set -euo pipefail
cd "$(dirname "$0")/.."

LOG="results/_launch_arch.log"
STATUS="results/status.json"
WD_KEYS='PathMNIST_100ep_seed42_wd0 PathMNIST_100ep_seed42_wd1e-05 PathMNIST_100ep_seed42_wd1e-03 PathMNIST_100ep_seed123_wd0 PathMNIST_100ep_seed123_wd1e-05 PathMNIST_100ep_seed123_wd1e-03 PathMNIST_100ep_seed2024_wd0 PathMNIST_100ep_seed2024_wd1e-05 PathMNIST_100ep_seed2024_wd1e-03'

echo "[$(date -Iseconds)] arch-launcher: waiting for WD sweep to finish" > "$LOG"

while true; do
    if [ ! -f "$STATUS" ]; then
        sleep 60; continue
    fi

    done_n=$(python3 - <<PY
import json
st = json.load(open("$STATUS"))
keys = "$WD_KEYS".split()
print(sum(1 for k in keys if st.get(k) == "completed"), end="")
PY
    )

    if pgrep -af "src.orchestrator.*experiment_grid_wd" > /dev/null; then
        wd_alive=yes
    else
        wd_alive=no
    fi

    echo "[$(date -Iseconds)] WD completed=$done_n/9  wd_orchestrator_alive=$wd_alive" >> "$LOG"

    # GPU free when WD sweep finished AND its orchestrator process is gone.
    if [ "$done_n" -ge 9 ] && [ "$wd_alive" = "no" ]; then
        echo "[$(date -Iseconds)] GPU free — launching Phase-3 (architecture sweep)" >> "$LOG"
        break
    fi

    # Safety: if Phase 2 died with partial runs, don't auto-launch Phase 3 —
    # leave it for human review.
    if [ "$wd_alive" = "no" ] && [ "$done_n" -lt 9 ]; then
        echo "[$(date -Iseconds)] WD orchestrator died with done=$done_n/9 — aborting auto-launch" >> "$LOG"
        echo "[$(date -Iseconds)] Investigate manually before launching Phase 3" >> "$LOG"
        exit 1
    fi

    sleep 60
done

# Brief settle delay so GPU memory is released and any final file writes flush.
sleep 5

# Launch Phase-3 orchestrator in background.
nohup ./.venv/bin/python -m src.orchestrator --config config/experiment_grid_arch.yaml \
    > results/_arch_orchestrator.log 2>&1 &
echo "[$(date -Iseconds)] launched arch orchestrator PID=$!" >> "$LOG"

# Launch Phase-3 watcher in background.
nohup ./scripts/watch_and_plot_arch.sh > /dev/null 2>&1 &
echo "[$(date -Iseconds)] launched arch watcher PID=$!" >> "$LOG"

echo "[$(date -Iseconds)] arch-launcher exiting" >> "$LOG"

#!/usr/bin/env bash
# campaign_heartbeat.sh -- run on the training pod alongside run_full_campaign.py
# (e.g. `nohup bash scripts/campaign_heartbeat.sh &`).
#
# Replaces the previous "sleep N_HOURS; curl -X DELETE" pod watchdog, which
# caused a real, total campaign-progress loss: a blind timer fired mid-run
# because the estimated duration turned out to be wrong (39-bus training is
# ~4x slower than assumed) and nobody revised it in time.
#
# This script does NOT delete or stop the pod, ever. It only observes and
# logs. Deciding to delete the pod is a deliberate action the operator (or
# the orchestrating agent, actively monitoring this log / campaign_status.json)
# takes after confirming the campaign actually finished and results are
# safe -- never something a background timer should do unattended.
#
# Usage: bash scripts/campaign_heartbeat.sh [status_json_path] [poll_seconds]
set -uo pipefail

STATUS_JSON="${1:-results/core_cache/../campaign_status.json}"
POLL_SECONDS="${2:-300}"
MAX_HOURS_BEFORE_LOUD_WARNING=336  # 14 days -- generous even for a pessimistic full-mode estimate

start_ts=$(date +%s)

echo "[heartbeat] watching ${STATUS_JSON} every ${POLL_SECONDS}s (this script never deletes the pod)"

while true; do
    now=$(date +%s)
    elapsed_h=$(( (now - start_ts) / 3600 ))

    if [ -f "$STATUS_JSON" ]; then
        last_phase=$(python3 -c "import json,sys; d=json.load(open('$STATUS_JSON')); print(d.get('last_phase','?'))" 2>/dev/null || echo "?")
        complete=$(python3 -c "import json,sys; d=json.load(open('$STATUS_JSON')); print(d.get('complete', False))" 2>/dev/null || echo "False")
    else
        last_phase="(no status file yet)"
        complete="False"
    fi

    echo "[heartbeat] t+${elapsed_h}h | last_phase=${last_phase} | complete=${complete}"

    if [ "$complete" = "True" ]; then
        echo "[heartbeat] campaign_status.json reports complete=true -- campaign finished."
        echo "[heartbeat] ACTION NEEDED: sync results off this pod, then delete it manually/via the orchestrator. This script will keep idly logging until you stop it."
    fi

    if [ "$elapsed_h" -ge "$MAX_HOURS_BEFORE_LOUD_WARNING" ]; then
        echo "[heartbeat] WARNING: ${elapsed_h}h elapsed with no completion -- this is far beyond even the pessimistic estimate. Check for a stuck job. This script still will NOT delete anything."
    fi

    sleep "$POLL_SECONDS"
done

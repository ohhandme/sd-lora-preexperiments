#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp/sd-lora-preexp/experiments/v2-routing
mkdir -p "$ROOT/logs" "$ROOT/status"
if [[ -f "$ROOT/status/fetch.pid" ]]; then
  old_pid=$(cat "$ROOT/status/fetch.pid")
  if kill -0 "$old_pid" 2>/dev/null; then
    echo "fetch already running pid=$old_pid"
    exit 0
  fi
fi
nohup /root/miniconda3/envs/spec/bin/python "$ROOT/scripts/fetch_pilot.py" \
  --sources "$ROOT/status/sources.json" \
  --root "$ROOT" \
  --out "$ROOT/status/fetch.json" \
  >"$ROOT/logs/fetch.log" 2>&1 </dev/null &
pid=$!
printf '%s\n' "$pid" >"$ROOT/status/fetch.pid"
echo "started fetch pid=$pid"

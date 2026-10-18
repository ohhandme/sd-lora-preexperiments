#!/usr/bin/env bash
set -euo pipefail
V2=/root/autodl-tmp/sd-lora-preexp/experiments/v2-routing
mkdir -p "$V2/logs" "$V2/status"
if [[ -f "$V2/status/coordinator.pid" ]]; then
  pid=$(cat "$V2/status/coordinator.pid")
  if kill -0 "$pid" 2>/dev/null; then
    echo "coordinator already running pid=$pid"
    exit 0
  fi
fi
nohup /root/miniconda3/envs/spec/bin/python "$V2/scripts/coordinator.py" \
  > "$V2/logs/coordinator.log" 2>&1 </dev/null &
pid=$!
printf '%s\n' "$pid" > "$V2/status/coordinator.pid"
echo "started coordinator pid=$pid"

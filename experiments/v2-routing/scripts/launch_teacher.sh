#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp/sd-lora-preexp
V2="$ROOT/experiments/v2-routing"
mkdir -p "$V2/logs" "$V2/status"
if [[ -f "$V2/status/teacher.pid" ]]; then
  old_pid=$(cat "$V2/status/teacher.pid")
  if kill -0 "$old_pid" 2>/dev/null; then
    echo "teacher already running pid=$old_pid"
    exit 0
  fi
fi
nohup flock -n "$ROOT/manifests/gpu_pipeline.lock" \
  /root/miniconda3/envs/spec/bin/python "$V2/scripts/teacher.py" --batch 8 \
  > "$V2/logs/teacher.log" 2>&1 </dev/null &
pid=$!
printf '%s\n' "$pid" > "$V2/status/teacher.pid"
echo "started v2 pilot teacher pid=$pid"

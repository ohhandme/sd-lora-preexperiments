#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp/sd-lora-preexp
V2="$ROOT/experiments/v2-routing"
if [[ -f "$V2/status/teacher.pid" ]]; then
  old_pid=$(cat "$V2/status/teacher.pid")
  if kill -0 "$old_pid" 2>/dev/null; then
    echo "old teacher still running pid=$old_pid; wait for batch-boundary pause"
    exit 1
  fi
fi
for index in 0 1; do
  pidfile="$V2/status/teacher.$index.pid"
  if [[ -f "$pidfile" ]]; then
    old_pid=$(cat "$pidfile")
    if kill -0 "$old_pid" 2>/dev/null; then
      echo "teacher shard $index already running pid=$old_pid"
      continue
    fi
  fi
  nohup flock -s -w 60 "$ROOT/manifests/gpu_pipeline.lock" \
    /root/miniconda3/envs/spec/bin/python "$V2/scripts/teacher.py" \
    --batch 8 --shard-index "$index" --shard-count 2 \
    > "$V2/logs/teacher.$index.log" 2>&1 </dev/null &
  pid=$!
  printf '%s\n' "$pid" > "$pidfile"
  echo "started teacher shard=$index pid=$pid"
done

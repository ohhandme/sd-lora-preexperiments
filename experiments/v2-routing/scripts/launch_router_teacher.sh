#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp/sd-lora-preexp
V2="$ROOT/experiments/v2-routing"
test -f "$V2/status/router_owner.json"
for index in 0 1; do
  pidfile="$V2/status/router_teacher.$index.pid"
  if [[ -f "$pidfile" ]]; then
    pid=$(cat "$pidfile")
    if kill -0 "$pid" 2>/dev/null; then
      echo "router teacher shard $index already running pid=$pid"
      continue
    fi
  fi
  nohup flock -s -w 120 "$ROOT/manifests/gpu_pipeline.lock" \
    /root/miniconda3/envs/spec/bin/python "$V2/scripts/router_teacher.py" \
    --batch 8 --shard-index "$index" --shard-count 2 \
    > "$V2/logs/router_teacher.$index.log" 2>&1 </dev/null &
  pid=$!
  printf '%s\n' "$pid" > "$pidfile"
  echo "started router teacher shard=$index pid=$pid"
done

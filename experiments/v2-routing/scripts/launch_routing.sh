#!/usr/bin/env bash
set -euo pipefail
V2=/root/autodl-tmp/sd-lora-preexp/experiments/v2-routing
ROOT=/root/autodl-tmp/sd-lora-preexp
PY=/root/miniconda3/envs/spec/bin/python
cd "$V2"
test "$("$PY" -c 'import json; x=json.load(open("results/quality/summary.json")); print(x["state"] == "complete" and x["completed_entries"] == 1920 and x["quality_headroom_only"])')" = True
if test -f results/routing/summary.json; then
  echo already_complete
  exit 0
fi
if test -f status/routing_launcher.pid && kill -0 "$(cat status/routing_launcher.pid)" 2>/dev/null; then
  echo already_running
  exit 0
fi
if pgrep -f '[s]cripts/routing.py' >/dev/null; then
  echo already_running
  exit 0
fi
nohup flock -w 120 "$ROOT/manifests/gpu_pipeline.lock" "$PY" "$V2/scripts/routing.py" > logs/worker.routing.log 2>&1 < /dev/null &
pid=$!
echo "$pid" > status/routing_launcher.pid
echo "started:$pid"

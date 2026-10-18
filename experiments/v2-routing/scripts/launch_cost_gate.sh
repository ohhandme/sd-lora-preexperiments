#!/usr/bin/env bash
set -euo pipefail
V2=/root/autodl-tmp/sd-lora-preexp/experiments/v2-routing
ROOT=/root/autodl-tmp/sd-lora-preexp
PY=/root/miniconda3/envs/spec/bin/python
cd "$V2"
"$PY" -m py_compile scripts/cost_gate.py
test -f results/routing/summary.json
if test -f results/cost_gate/summary.json; then
  echo already_complete
  exit 0
fi
if test -f status/cost_gate_launcher.pid && kill -0 "$(cat status/cost_gate_launcher.pid)" 2>/dev/null; then
  echo already_running
  exit 0
fi
if pgrep -f '[s]cripts/cost_gate.py' >/dev/null; then
  echo already_running
  exit 0
fi
nohup flock -w 120 "$ROOT/manifests/gpu_pipeline.lock" "$PY" "$V2/scripts/cost_gate.py" > logs/worker.cost_gate.log 2>&1 < /dev/null &
pid=$!
echo "$pid" > status/cost_gate_launcher.pid
echo "started:$pid"

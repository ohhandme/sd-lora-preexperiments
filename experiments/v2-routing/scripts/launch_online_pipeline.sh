#!/usr/bin/env bash
set -euo pipefail
V2=/root/autodl-tmp/sd-lora-preexp/experiments/v2-routing
ROOT=/root/autodl-tmp/sd-lora-preexp
PY=/root/miniconda3/envs/spec/bin/python
cd "$V2"
test "$("$PY" -c 'import json; x=json.load(open("results/cost_gate/summary.json")); print(x["state"] == "complete" and x["optimistic_hardware_screen"])')" = True
test "$("$PY" -c 'import json; x=json.load(open("results/routing/summary.json")); print(x["state"] == "complete")')" = True
if test -f results/online_pilot/summary.json; then
  echo already_complete
  exit 0
fi
if test -f status/online_pipeline.pid && kill -0 "$(cat status/online_pipeline.pid)" 2>/dev/null; then
  echo already_running
  exit 0
fi
if pgrep -f '[s]cripts/online.py --method' >/dev/null; then
  echo already_running
  exit 0
fi
nohup bash -c '
  set -euo pipefail
  cd /root/autodl-tmp/sd-lora-preexp/experiments/v2-routing
  for method in AR8 BASE mix16 mix80 DOMAIN ROUTE-TASK ROUTE-ACC; do
    echo "START $method $(date -u +%FT%TZ)" >> logs/online_pipeline.log
    flock -w 120 /root/autodl-tmp/sd-lora-preexp/manifests/gpu_pipeline.lock /root/miniconda3/envs/spec/bin/python scripts/online.py --method "$method" > "logs/online.$method.log" 2>&1
    echo "DONE $method $(date -u +%FT%TZ)" >> logs/online_pipeline.log
  done
  /root/miniconda3/envs/spec/bin/python scripts/online_report.py > logs/online_report.log 2>&1
  echo "REPORT_COMPLETE $(date -u +%FT%TZ)" >> logs/online_pipeline.log
' > logs/online_pipeline.stdout.log 2>&1 < /dev/null &
pid=$!
echo "$pid" > status/online_pipeline.pid
echo "started:$pid"

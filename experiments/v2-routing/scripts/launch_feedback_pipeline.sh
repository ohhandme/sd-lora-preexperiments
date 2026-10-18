#!/usr/bin/env bash
set -euo pipefail
V2=/root/autodl-tmp/sd-lora-preexp/experiments/v2-routing
cd "$V2"
if test -f results/online_pilot/feedback_summary.json; then
  echo already_complete
  exit 0
fi
if test -f status/feedback_pipeline.pid && kill -0 "$(cat status/feedback_pipeline.pid)" 2>/dev/null; then
  echo already_running
  exit 0
fi
nohup bash -c '
  set -euo pipefail
  cd /root/autodl-tmp/sd-lora-preexp/experiments/v2-routing
  while ! test -f results/online_pilot/summary.json; do
    if ! kill -0 "$(cat status/online_pipeline.pid)" 2>/dev/null; then
      echo "STATIC_PIPELINE_STOPPED $(date -u +%FT%TZ)" >> logs/feedback_pipeline.log
      exit 13
    fi
    sleep 30
  done
  for method in META-UCB COST-UCB; do
    if ! test -f results/smoke/feedback_$method.json; then
      echo "SMOKE_START $method $(date -u +%FT%TZ)" >> logs/feedback_pipeline.log
      flock -w 120 /root/autodl-tmp/sd-lora-preexp/manifests/gpu_pipeline.lock /root/miniconda3/envs/spec/bin/python scripts/feedback_online.py --method "$method" --smoke > "logs/feedback.smoke.$method.log" 2>&1
      echo "SMOKE_DONE $method $(date -u +%FT%TZ)" >> logs/feedback_pipeline.log
    fi
    echo "START $method $(date -u +%FT%TZ)" >> logs/feedback_pipeline.log
    flock -w 120 /root/autodl-tmp/sd-lora-preexp/manifests/gpu_pipeline.lock /root/miniconda3/envs/spec/bin/python scripts/feedback_online.py --method "$method" > "logs/feedback.$method.log" 2>&1
    echo "DONE $method $(date -u +%FT%TZ)" >> logs/feedback_pipeline.log
  done
  /root/miniconda3/envs/spec/bin/python scripts/feedback_report.py > logs/feedback_report.log 2>&1
  echo "REPORT_COMPLETE $(date -u +%FT%TZ)" >> logs/feedback_pipeline.log
' > logs/feedback_pipeline.stdout.log 2>&1 < /dev/null &
pid=$!
echo "$pid" > status/feedback_pipeline.pid
echo "started:$pid"

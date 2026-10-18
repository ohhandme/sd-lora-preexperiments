#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/sd-lora-preexp
base=experiments/v3-async-budget
python_bin=/root/miniconda3/envs/spec/bin/python
mkdir -p "$base/logs" "$base/results/online_diagnostic"
for method in AR L8 H4-8; do
  now=$(date -u +%s)
  deadline=$(date -u -d '2026-10-18 09:00:00 UTC' +%s)
  if (( now >= deadline )); then
    printf 'Deadline reached before %s\n' "$method" > "$base/results/online_diagnostic/deadline_stop.txt"
    exit 1
  fi
  timeout --signal=INT --kill-after=30s 45m \
    "$python_bin" "$base/scripts/online_diagnostic.py" --method "$method" \
    >> "$base/logs/online_diagnostic.log" 2>&1
done
printf 'complete %s\n' "$(date -u -Is)" > "$base/results/online_diagnostic/launcher_complete.txt"

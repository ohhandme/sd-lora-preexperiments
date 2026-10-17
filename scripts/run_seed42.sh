#!/bin/bash
set -euo pipefail
cd /root/autodl-tmp/sd-lora-preexp
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec /root/miniconda3/envs/spec/bin/python -u run_seed42_pipeline.py --run-id "${1:?immutable run ID required}"

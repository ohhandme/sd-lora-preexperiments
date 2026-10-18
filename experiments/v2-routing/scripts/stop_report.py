"""Finalize the frozen v2 pilot after the preregistered P3 negative gate."""

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

root = Path(__file__).resolve().parents[3]
v2 = root / "experiments" / "v2-routing"


def read(rel):
    return json.loads((v2 / rel).read_text())


audit = read("results/final_audit.json")
assert audit["state"] == "passed" and audit["full_token_eos_truncation_mismatches"] == 0
static = read("results/online_pilot/summary.json")
feedback = read("results/online_pilot/feedback_summary.json")
quality = read("results/quality/summary.json")
routing = read("results/routing/summary.json")
cost = read("results/cost_gate/summary.json")
preflight = read("status/preflight.json")
teacher = [read(f"status/teacher.{i}.json") for i in (0, 1)]
router_teacher = [read(f"status/router_teacher.{i}.json") for i in (0, 1)]
trains = {m: read(f"status/train.{m}.json") for m in ("code", "translation", "summary", "qa", "math", "mix16", "mix80")}
assert preflight["state"] == "passed"
assert sum(x["completed"] for x in teacher) == 1520
assert sum(x["completed"] for x in router_teacher) == 600
assert all(x["state"] == "complete" for x in teacher + router_teacher)
assert all(x["state"] == "complete" for x in trains.values())
assert static["state"] == feedback["state"] == "complete"

oracle_mix = quality["paired"]["ORACLE_minus_mix80"]
oracle_fixed = quality["paired"]["ORACLE_minus_FIXEDBEST"]
assert oracle_mix["ci95"][0] > 0 and oracle_fixed["ci95"][0] > 0
candidate = feedback["methods"]["COST-UCB"]
meta = feedback["methods"]["META-UCB"]
assert candidate["speedup_vs_MIX80_ci95"][1] < 1.0
assert candidate["speedup_vs_AR_ci95"][1] < 1.0
gpu_process_lines = subprocess.check_output(
    ["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader"],
    text=True,
).strip()
assert not gpu_process_lines, f"GPU compute processes remain: {gpu_process_lines}"

paths = [
    "results/final_audit.json",
    "results/quality/summary.json",
    "results/routing/summary.json",
    "results/cost_gate/summary.json",
    "results/online_pilot/summary.json",
    "results/online_pilot/feedback_summary.json",
    "results/online_pilot/subset_manifest.json",
]
artifacts = {str(v2 / p): hashlib.sha256((v2 / p).read_bytes()).hexdigest() for p in paths}

report = {
    "at": datetime.now(timezone.utc).isoformat(),
    "state": "completed_with_preregistered_negative_P3_stop",
    "protocol": "v2.0 section 2; five-task 0.6B multi-LoRA pilot; seed42",
    "program_sha256": audit["program_hashes_checked"][str(v2 / "program.md")],
    "pilot_manifest_sha256": "9542965d3fd3f1dda33cc189978576a65e61d24676a1b511f45f886ea43fede7",
    "gate_outcomes": {
        "P0v2": {"state": "passed", "evidence": "pinned assets, isolated pilot, BF16/tokenizer gate, batch8-vs-single full-token audits, separate v1/v2 hashes"},
        "P1": {"state": "complete", "teacher_records": 2120, "trained_selected_adapters": 7, "quality_entries": 1920, "quality_requests": 240, "quality_qualified_requests": 238},
        "P2": {"state": "offline_headroom_positive_hardware_screen_only", "oracle_minus_mix80_accepted_tokens": oracle_mix, "oracle_minus_fixedbest_accepted_tokens": oracle_fixed, "prompt_route_minus_mix80_val_tokens": routing["mean_a"]["acc_score"] - routing["mean_a"]["mix80"], "switch_ms_median": cost["adapter_switch_ms_median"], "router_ms_mean": cost["prompt_router_ms_mean"], "caveat": "Oracle uses future target outputs. The hardware screen is optimistic and does not establish a feasible speed gain."},
        "P3": {"state": "negative", "holdout_correctness": "40 router_val requests x 8 SD methods = 320 exact token/EOS/truncation matches, zero mismatch", "candidate": "COST-UCB", "candidate_geomean_vs_AR": candidate["speedup_vs_AR_geomean"], "candidate_ci95_vs_AR": candidate["speedup_vs_AR_ci95"], "candidate_geomean_vs_MIX80": candidate["speedup_vs_MIX80_geomean"], "candidate_ci95_vs_MIX80": candidate["speedup_vs_MIX80_ci95"], "meta_ucb_geomean_vs_MIX80": meta["speedup_vs_MIX80_geomean"], "static_mix80_geomean_vs_AR": static["methods"]["mix80"]["speedup_geomean"], "static_base_geomean_vs_AR": static["methods"]["BASE"]["speedup_geomean"], "reason": "Candidate is slower than equal-data MIX80 and target AR on the paired pilot; its upper bootstrap bounds remain below 1.0. Stop under section 2.5 without unlocking P4."},
        "P4": {"state": "not_unlocked", "reason": "P3 negative gate", "unrun": ["full 5-task training set", "seed43/44", "three randomized timing repeats", "locked final ID/external tasks", "real WAN edge-cloud measurement", "4B five-expert training"]},
    },
    "measurement_scope": {"GPU": "NVIDIA RTX 6000D, single card", "L": 8, "timing_repeats": 1, "independent_AR_requests": 40, "paired_SD_comparisons": 320, "router_val_online_requests_per_method": 40},
    "interpretation": "A retrospective five-expert Oracle has acceptance headroom, but neither static prompt routing nor this MetaSD-inspired UCB baseline nor the tested cost-constrained feedback candidate converts it into local speedup. This is a negative pilot for the current implementation/configuration, not a universal refutation of speculative decoding or edge-cloud pipelining.",
    "possible_decoder_bottleneck": "All SD methods including the untuned BASE are slower than target AR in this Python cached BF16 single-card implementation. This is compatible with decoder/Python/kernel overhead and sequential target/draft costs; no dedicated component CUDA profile isolates the cause, so it is not claimed as proven.",
    "limitations": ["40 requests (8 per task), one timing repeat", "No independent training seed replication", "NQ-Open is not full Natural Questions", "Online holdout is disjoint from training/smoke but final ID/external test remains locked", "Meta-UCB is MetaSD-inspired, not faithful MetaSD reproduction", "Oracle is offline only", "No true WAN/edge-cloud measurement"],
    "artifacts_sha256": artifacts,
    "gpu_compute_processes_at_finalize": [],
}
out = v2 / "results" / "final_report.json"
tmp = out.with_suffix(".json.tmp")
tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
tmp.replace(out)
print(json.dumps({"state": report["state"], "path": str(out), "candidate_vs_mix80": candidate["speedup_vs_MIX80_geomean"]}, ensure_ascii=False))

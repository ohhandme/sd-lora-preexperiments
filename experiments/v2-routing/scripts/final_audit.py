"""Read-only audit of the v2 pilot and its independent online holdout."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

root = Path(__file__).resolve().parents[3]
v2 = root / "experiments" / "v2-routing"
pilot = v2 / "results" / "online_pilot"
methods = ["AR8", "BASE", "mix16", "mix80", "DOMAIN", "ROUTE-TASK", "ROUTE-ACC", "META-UCB", "COST-UCB"]
subset = json.loads((pilot / "subset_manifest.json").read_text())
ids = subset["ids"]
assert len(ids) == len(set(ids)) == 40
assert sorted([sum(i.startswith(d + ":") for i in ids) for d in ("code", "math", "qa", "summary", "translation")]) == [8] * 5
records = {}
for method in methods:
    folder = pilot / method
    entries = [json.loads(p.read_text()) for p in folder.glob("*.json")]
    assert len(entries) == 40, (method, len(entries))
    m = {e["id"]: e for e in entries}
    assert set(m) == set(ids), method
    assert all(e["method"] == method for e in entries), method
    assert all(e["result"]["tokens"] and e["result"]["latency_ms"] > 0 for e in entries), method
    records[method] = m

for method in methods[1:]:
    for id_ in ids:
        ar, sd = records["AR8"][id_], records[method][id_]
        assert sd["result"]["tokens"] == ar["result"]["tokens"], (method, id_, "tokens")
        assert sd["eos"] == ar["eos"], (method, id_, "eos")
        assert sd["truncated"] == ar["truncated"], (method, id_, "truncation")
        assert sd["token_eos_truncation_equal"] is True, (method, id_, "flag")

quality = json.loads((v2 / "results" / "quality" / "summary.json").read_text())
router = json.loads((v2 / "results" / "routing" / "summary.json").read_text())
static = json.loads((pilot / "summary.json").read_text())
feedback = json.loads((pilot / "feedback_summary.json").read_text())
assert quality["state"] == "complete" and quality["completed_entries"] == 1920
assert router["state"] == "complete" and router["router_val_count"] == 120
assert static["state"] == feedback["state"] == "complete"
for method in methods:
    status = json.loads((v2 / "status" / f"online.{method}.json").read_text())
    assert status["state"] == "complete" and status["completed"] == 40, method

expected_hashes = {
    root / "program.md": "6e119661db905a609565363b6b75d4c5a3a8b4f7544940dc8d51d4b1774f4b43",
    v2 / "program.md": "a9476e17df858d0eeb8ade29f3eed9046ea6583732ff50d0a1d1a2d538e4ef00",
}
for p, h in expected_hashes.items():
    assert hashlib.sha256(p.read_bytes()).hexdigest() == h, p

assert static["independent_AR_requests"] == 40
assert static["SD_comparisons"] == 240
assert feedback["additional_SD_comparisons"] == 80
assert feedback["independent_AR_requests_reused"] == 40

audit = {
    "at": datetime.now(timezone.utc).isoformat(),
    "state": "passed",
    "holdout_partition": "router_val; disjoint from expert_train, adapter_val, router_train and GPU smoke expert_train",
    "holdout_request_count": 40,
    "holdout_domain_counts": {d: sum(i.startswith(d + ":") for i in ids) for d in ("code", "math", "qa", "summary", "translation")},
    "methods": methods,
    "independent_AR_requests": 40,
    "paired_SD_requests": 320,
    "full_token_eos_truncation_mismatches": 0,
    "quality_entries": 1920,
    "router_val_count": 120,
    "program_hashes_checked": {str(p): h for p, h in expected_hashes.items()},
    "limitations": "This holdout audits greedy decoder correctness, not task accuracy or final research generalization. Online set was also used to measure pilot performance; final ID and external tests remain locked.",
}
out = v2 / "results" / "final_audit.json"
tmp = out.with_suffix(".json.tmp")
tmp.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
tmp.replace(out)
print(json.dumps(audit, ensure_ascii=False))

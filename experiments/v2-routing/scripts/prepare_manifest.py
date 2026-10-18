#!/root/miniconda3/envs/spec/bin/python
"""Freeze v2 five-task train/validation IDs before any teacher generation."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
import pathlib
from collections import Counter
from datetime import datetime, timezone

import pyarrow.parquet as parquet
from transformers import AutoTokenizer


TARGET_PER_DOMAIN = {
    "expert_train": 2000,
    "adapter_val": 200,
    "router_train": 400,
    "router_val": 100,
}
PILOT_PER_DOMAIN = {
    "expert_train": 256,
    "adapter_val": 48,
    "router_train": 96,
    "router_val": 24,
}
SEED = "20261017-v2-five-task-v1"


def stable_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def atomic_text(path: pathlib.Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def source_rows(root: pathlib.Path, domain: str):
    paths = {
        "math": root / "data/raw/math/gsm8k_main/main/train-00000-of-00001.parquet",
        "code": root
        / "data/raw/code/magicoder_oss_instruct_75k/data-oss_instruct-decontaminated.jsonl",
        "summary": root
        / "experiments/v2-routing/data/raw/summary/train-00000-of-00003.parquet",
        "translation": root
        / "experiments/v2-routing/data/raw/translation/train-00000-of-00003.parquet",
        "qa": root
        / "experiments/v2-routing/data/raw/qa/train-00000-of-00001.parquet",
    }
    path = paths[domain]
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            for row_number, line in enumerate(handle):
                yield row_number, json.loads(line)
    else:
        source = parquet.ParquetFile(path)
        row_number = 0
        for batch in source.iter_batches(batch_size=512):
            for record in batch.to_pylist():
                yield row_number, record
                row_number += 1


def build_prompt(domain: str, record: dict) -> str | None:
    if domain == "math":
        content = record.get("question")
        return f"Solve the following math problem.\n\n{content}" if content else None
    if domain == "code":
        if str(record.get("lang", "")).lower() != "python":
            return None
        content = record.get("problem")
        return f"Write Python code for this task.\n\n{content}" if content else None
    if domain == "summary":
        content = record.get("article")
        return f"Summarize the following article.\n\n{content}" if content else None
    if domain == "translation":
        pair = record.get("translation") or {}
        content = pair.get("de")
        return f"Translate the following German text into English.\n\n{content}" if content else None
    if domain == "qa":
        content = record.get("question")
        return f"Answer the following question.\n\n{content}" if content else None
    raise ValueError(domain)


def source_id(domain: str, index: int, record: dict) -> str:
    if domain == "code":
        return str(record.get("raw_index", index))
    if domain == "summary":
        return str(record.get("id", index))
    return str(index)


def collect_candidates(root: pathlib.Path, domain: str, capacity: int = 15000):
    # Keep the lowest deterministic hashes. Tokenizing only these candidates
    # bounds the cost for the 1.5M-row WMT training shard.
    heap: list[tuple[int, str, str, dict]] = []
    scanned = 0
    valid = 0
    for index, row in source_rows(root, domain):
        scanned += 1
        prompt = build_prompt(domain, row)
        if not prompt or not prompt.strip():
            continue
        valid += 1
        normalized = " ".join(prompt.split())
        prompt_hash = stable_digest(normalized)
        key = int(stable_digest(f"{SEED}|{domain}|{prompt_hash}"), 16)
        item = {
            "domain": domain,
            "source_id": source_id(domain, index, row),
            "source_row": index,
            "prompt": prompt,
            "prompt_sha256": prompt_hash,
            "split_key": f"{key:064x}",
        }
        entry = (-key, prompt_hash, item["source_id"], item)
        if len(heap) < capacity:
            heapq.heappush(heap, entry)
        elif entry > heap[0]:
            heapq.heapreplace(heap, entry)
    return [item for _, _, _, item in sorted(heap, key=lambda value: -value[0])], {
        "scanned_rows": scanned,
        "candidate_rows": valid,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    assets = json.loads((root / "configs/assets.local.json").read_text())
    sources = json.loads(
        (root / "experiments/v2-routing/status/sources.json").read_text()
    )
    fetch = json.loads((root / "experiments/v2-routing/status/fetch.json").read_text())
    if sources["state"] != "resolved" or fetch["state"] != "complete":
        raise SystemExit("all selected source files must be verified first")
    tokenizer = AutoTokenizer.from_pretrained(
        root / "models/target/Qwen3-8B",
        local_files_only=True,
        trust_remote_code=False,
    )
    system_prompt = json.loads((root / 'configs/protocol.json').read_text())['system']
    total_needed = sum(TARGET_PER_DOMAIN.values())
    full_rows: list[dict] = []
    pilot_rows: list[dict] = []
    stats: dict = {}
    global_prompt_hashes: set[str] = set()
    for domain in ("math", "code", "summary", "translation", "qa"):
        candidates, scanned = collect_candidates(root, domain)
        filtered = Counter()
        selected = []
        for item in candidates:
            prompt_hash = item["prompt_sha256"]
            if prompt_hash in global_prompt_hashes:
                filtered["duplicate_input"] += 1
                continue
            ids = tokenizer.apply_chat_template(
                [{'role': 'system', 'content': system_prompt},
                 {'role': 'user', 'content': item['prompt']}],
                tokenize=True, add_generation_prompt=True, enable_thinking=False,
            )
            if len(ids) > 1024:
                filtered["over_1024_tokens"] += 1
                continue
            if not ids:
                filtered["empty_tokens"] += 1
                continue
            item["prompt_tokens"] = len(ids)
            selected.append(item)
            global_prompt_hashes.add(prompt_hash)
            if len(selected) == total_needed:
                break
        if len(selected) != total_needed:
            raise RuntimeError(f"{domain} has {len(selected)} < {total_needed} valid rows")
        if domain in ("math", "code"):
            source_record = assets["datasets"][
                "openai/gsm8k"
                if domain == "math"
                else "ise-uiuc/Magicoder-OSS-Instruct-75K"
            ]
            revision = source_record["revision"]
        else:
            revision = sources["sources"][domain]["revision"]
        offset = 0
        counts = {}
        for partition, size in TARGET_PER_DOMAIN.items():
            block = selected[offset : offset + size]
            counts[partition] = len(block)
            for position, item in enumerate(block):
                item["partition"] = partition
                item["source_revision"] = revision
                item["record_id"] = f"{domain}:{item['source_id']}:{item['prompt_sha256']}"
                full_rows.append(item)
                if position < PILOT_PER_DOMAIN[partition]:
                    pilot_rows.append(item)
            offset += size
        stats[domain] = {**scanned, "filter_counts": dict(filtered), "counts": counts}
    if len({row["record_id"] for row in full_rows}) != len(full_rows):
        raise RuntimeError("duplicate record IDs after split")
    if len({row["prompt_sha256"] for row in full_rows}) != len(full_rows):
        raise RuntimeError("cross-partition prompt leakage")
    if len(pilot_rows) != 5 * sum(PILOT_PER_DOMAIN.values()):
        raise RuntimeError("pilot count mismatch")
    args.out.mkdir(parents=True, exist_ok=True)
    full_path = args.out / "train_val_manifest.jsonl"
    pilot_path = args.out / "pilot_manifest.jsonl"
    atomic_text(
        full_path,
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in full_rows),
    )
    atomic_text(
        pilot_path,
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in pilot_rows),
    )
    manifest = {
        "at": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "state": "frozen",
        "full_count": len(full_rows),
        "pilot_count": len(pilot_rows),
        "full_sha256": hashlib.sha256(full_path.read_bytes()).hexdigest(),
        "pilot_sha256": hashlib.sha256(pilot_path.read_bytes()).hexdigest(),
        "partition_counts": TARGET_PER_DOMAIN,
        "pilot_partition_counts": PILOT_PER_DOMAIN,
        "domain_stats": stats,
        "test_inputs_read": False,
        "note": "Official held-out files are registered but not opened or used for selection.",
    }
    atomic_text(
        args.out / "manifest.json",
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

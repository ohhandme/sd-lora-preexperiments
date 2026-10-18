#!/root/miniconda3/envs/spec/bin/python
"""Read-only assets/GPU preflight for the isolated v2 routing experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import shutil
import sys
import time
from datetime import datetime, timezone


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: pathlib.Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    result: dict = {
        "at": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "interpreter": sys.executable,
        "state": "failed",
        "checks": {},
        "errors": [],
    }
    try:
        import datasets
        import peft
        import torch
        import transformers
        from transformers import AutoTokenizer

        result["versions"] = {
            "datasets": datasets.__version__,
            "peft": peft.__version__,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        }
        result["checks"]["spec_interpreter"] = (
            sys.executable == "/root/miniconda3/envs/spec/bin/python"
        )
        original_program = root / "program.md"
        result["v1_program_sha256"] = sha256(original_program)
        result["checks"]["v1_program_frozen"] = (
            result["v1_program_sha256"]
            == "6e119661db905a609565363b6b75d4c5a3a8b4f7544940dc8d51d4b1774f4b43"
        )
        new_program = root / "experiments/v2-routing/program.md"
        result["v2_program_sha256"] = sha256(new_program)
        result["checks"]["v2_program_distinct"] = (
            result["v2_program_sha256"] != result["v1_program_sha256"]
        )
        required_models = {
            "target": root / "models/target/Qwen3-8B",
            "draft_0_6B": root / "models/draft/Qwen3-0.6B",
            "draft_4B": root / "models/draft/Qwen3-4B",
        }
        for name, path in required_models.items():
            result["checks"][f"{name}_present"] = (
                path.is_dir() and (path / "config.json").is_file()
            )
        target_tokenizer = AutoTokenizer.from_pretrained(
            required_models["target"], local_files_only=True, trust_remote_code=False
        )
        draft_tokenizer = AutoTokenizer.from_pretrained(
            required_models["draft_0_6B"],
            local_files_only=True,
            trust_remote_code=False,
        )
        probes = [
            "A short math question: what is 17 + 25?",
            "Write a Python function to sort a list.",
            "Translate Guten Morgen into English.",
            "Summarize a short news article.",
            "Answer a question using only the given passage.",
        ]
        result["checks"]["tokenizer_probe_equal"] = all(
            target_tokenizer.encode(item, add_special_tokens=False)
            == draft_tokenizer.encode(item, add_special_tokens=False)
            for item in probes
        )
        result["checks"]["special_tokens_equal"] = (
            target_tokenizer.eos_token_id == draft_tokenizer.eos_token_id
            and target_tokenizer.pad_token_id == draft_tokenizer.pad_token_id
        )
        usage = shutil.disk_usage(root)
        result["disk"] = {
            "total_bytes": usage.total,
            "free_bytes": usage.free,
        }
        # Pilot assets are budgeted separately; this floor is only to prevent
        # starting writes on a nearly full filesystem.
        result["checks"]["pilot_disk_floor"] = usage.free >= 4_000_000_000
        result["checks"]["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            properties = torch.cuda.get_device_properties(device)
            free, total = torch.cuda.mem_get_info(device)
            result["gpu"] = {
                "name": properties.name,
                "total_bytes": properties.total_memory,
                "free_bytes": free,
                "cuda_mem_total_bytes": total,
            }
            first = torch.randn(128, 128, device=device, dtype=torch.bfloat16)
            second = torch.randn(128, 128, device=device, dtype=torch.bfloat16)
            product = first @ second
            torch.cuda.synchronize()
            result["checks"]["bf16_forward"] = bool(
                torch.isfinite(product).all().item()
            )
            import torch.nn.functional as functional

            query = torch.randn(1, 2, 16, 32, device=device, dtype=torch.bfloat16)
            output = functional.scaled_dot_product_attention(query, query, query)
            torch.cuda.synchronize()
            result["checks"]["sdpa_forward"] = bool(
                torch.isfinite(output).all().item()
            )
        result["state"] = (
            "passed" if all(result["checks"].values()) else "failed"
        )
    except Exception as error:
        result["errors"].append(f"{type(error).__name__}: {error}")
    write_json_atomic(args.out, result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["state"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

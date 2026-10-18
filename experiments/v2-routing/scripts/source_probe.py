#!/root/miniconda3/envs/spec/bin/python
"""Resolve v2 dataset revisions and parquet split files without bulk downloads."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from datetime import datetime, timezone

from huggingface_hub import HfApi


DATASETS = {
    "summary": ("abisee/cnn_dailymail", "3.0.0/"),
    "translation": ("wmt/wmt16", "de-en/"),
    "qa": ("google-research-datasets/nq_open", ""),
    "code_eval": ("HuggingFaceH4/mt_bench_prompts", ""),
}


def atomic_json(path: pathlib.Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()
    api = HfApi(endpoint="https://hf-mirror.com")
    result: dict = {
        "at": datetime.now(timezone.utc).isoformat(),
        "interpreter": sys.executable,
        "sources": {},
        "state": "running",
    }
    for domain, (dataset_id, path_prefix) in DATASETS.items():
        record = {"dataset_id": dataset_id, "path_prefix": path_prefix}
        try:
            info = api.dataset_info(dataset_id, files_metadata=True, timeout=15)
            record["revision"] = info.sha
            record["files"] = [
                {
                    "path": item.rfilename,
                    "size": item.size,
                    "lfs_sha256": (
                        item.lfs.get("sha256") if item.lfs is not None else None
                    ),
                }
                for item in info.siblings
                if item.rfilename.startswith(path_prefix)
                and (
                    item.rfilename.endswith(".parquet")
                    or item.rfilename.endswith(".jsonl")
                    or item.rfilename.endswith(".json")
                )
            ]
            record["state"] = "resolved" if record["files"] else "no_data_files"
        except Exception as error:
            record["state"] = "error"
            record["error"] = f"{type(error).__name__}: {error}"
        result["sources"][domain] = record
        atomic_json(args.out, result)
    result["state"] = (
        "resolved"
        if all(item["state"] == "resolved" for item in result["sources"].values())
        else "needs_review"
    )
    atomic_json(args.out, result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["state"] == "resolved" else 1


if __name__ == "__main__":
    raise SystemExit(main())

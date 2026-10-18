#!/root/miniconda3/envs/spec/bin/python
"""Fetch only the pinned v2 pilot shards, with resumable atomic files."""

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
from urllib.parse import quote

import requests


SELECTED = {
    "code_eval": ("data/train-00000-of-00001-67c6c9fef07685a3.parquet",),
    "qa": (
        "nq_open/train-00000-of-00001.parquet",
        "nq_open/validation-00000-of-00001.parquet",
    ),
    "summary": (
        "3.0.0/train-00000-of-00003.parquet",
        "3.0.0/test-00000-of-00001.parquet",
    ),
    "translation": (
        "de-en/train-00000-of-00003.parquet",
        "de-en/test-00000-of-00001.parquet",
    ),
}
MIN_FREE_BYTES = 4_000_000_000


def digest(path: pathlib.Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def atomic_status(path: pathlib.Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def fetch_one(
    session: requests.Session,
    repo: str,
    revision: str,
    item: dict,
    destination: pathlib.Path,
    result: dict,
    status_path: pathlib.Path,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected_size = item["size"]
    expected_digest = item["lfs_sha256"]
    if destination.exists():
        if destination.stat().st_size == expected_size and (
            not expected_digest or digest(destination) == expected_digest
        ):
            result["files"][str(destination)] = {
                "state": "verified",
                "size": expected_size,
                "sha256": digest(destination),
                "source": item["path"],
            }
            atomic_status(status_path, result)
            return
        raise RuntimeError(f"existing final file differs: {destination}")
    partial = destination.with_name(destination.name + ".part")
    url = (
        f"https://hf-mirror.com/datasets/{repo}/resolve/{revision}/"
        f"{quote(item['path'], safe='/')}?download=true"
    )
    for attempt in range(3):
        current = partial.stat().st_size if partial.exists() else 0
        if current > expected_size:
            raise RuntimeError(f"partial file exceeds expected size: {partial}")
        headers = {"Range": f"bytes={current}-"} if current else {}
        try:
            with session.get(url, headers=headers, stream=True, timeout=(15, 60)) as response:
                response.raise_for_status()
                if current and response.status_code == 200:
                    partial.unlink()
                    current = 0
                elif current and response.status_code != 206:
                    raise RuntimeError(
                        f"range resume returned HTTP {response.status_code}"
                    )
                mode = "ab" if current else "wb"
                with partial.open(mode) as handle:
                    last_recorded = time.monotonic()
                    for block in response.iter_content(chunk_size=1024 * 1024):
                        if not block:
                            continue
                        handle.write(block)
                        now = time.monotonic()
                        if now - last_recorded > 10:
                            result["current"] = {
                                "file": item["path"],
                                "downloaded_bytes": handle.tell(),
                                "expected_bytes": expected_size,
                            }
                            atomic_status(status_path, result)
                            last_recorded = now
            if partial.stat().st_size != expected_size:
                raise RuntimeError(
                    f"size mismatch {partial.stat().st_size} != {expected_size}"
                )
            actual = digest(partial)
            if expected_digest and actual != expected_digest:
                raise RuntimeError(f"sha256 mismatch for {item['path']}")
            os.replace(partial, destination)
            result["files"][str(destination)] = {
                "state": "verified",
                "size": expected_size,
                "sha256": actual,
                "source": item["path"],
                "repo": repo,
                "revision": revision,
            }
            result.pop("current", None)
            atomic_status(status_path, result)
            return
        except requests.RequestException as error:
            result["last_retryable_error"] = f"{type(error).__name__}: {error}"
            atomic_status(status_path, result)
            if attempt == 2:
                raise
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"exhausted retries for {item['path']}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=pathlib.Path, required=True)
    parser.add_argument("--root", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()
    sources = json.loads(args.sources.read_text(encoding="utf-8"))
    if sources["state"] != "resolved":
        raise SystemExit("source revisions are not fully resolved")
    result: dict = {
        "at": datetime.now(timezone.utc).isoformat(),
        "interpreter": sys.executable,
        "state": "running",
        "files": {},
    }
    try:
        selected = []
        for domain, paths in SELECTED.items():
            source = sources["sources"][domain]
            files = {item["path"]: item for item in source["files"]}
            for path in paths:
                selected.append((domain, source, files[path]))
        required = sum(item["size"] for _, _, item in selected)
        free = shutil.disk_usage(args.root).free
        result["estimated_download_bytes"] = required
        result["free_bytes_before"] = free
        if free - required < MIN_FREE_BYTES:
            raise RuntimeError("pilot files would cross reserved free-space floor")
        atomic_status(args.out, result)
        session = requests.Session()
        for domain, source, item in selected:
            destination = args.root / "data/raw" / domain / pathlib.Path(item["path"]).name
            fetch_one(
                session,
                source["dataset_id"],
                source["revision"],
                item,
                destination,
                result,
                args.out,
            )
        result["state"] = "complete"
        result["free_bytes_after"] = shutil.disk_usage(args.root).free
    except Exception as error:
        result["state"] = "failed"
        result["error"] = f"{type(error).__name__}: {error}"
    atomic_status(args.out, result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["state"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

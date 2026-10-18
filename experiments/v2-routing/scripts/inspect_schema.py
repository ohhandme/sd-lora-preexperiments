#!/root/miniconda3/envs/spec/bin/python
"""Print only dataset columns and one shortened representative record per source."""

from __future__ import annotations

import json
import pathlib

import pyarrow.parquet as parquet


ROOT = pathlib.Path("/root/autodl-tmp/sd-lora-preexp")
SOURCES = {
    "math": ROOT / "data/raw/math/gsm8k_main/main/train-00000-of-00001.parquet",
    "code": ROOT
    / "data/raw/code/magicoder_oss_instruct_75k/data-oss_instruct-decontaminated.jsonl",
    "summary": ROOT
    / "experiments/v2-routing/data/raw/summary/train-00000-of-00003.parquet",
    "translation": ROOT
    / "experiments/v2-routing/data/raw/translation/train-00000-of-00003.parquet",
    "qa": ROOT / "experiments/v2-routing/data/raw/qa/train-00000-of-00001.parquet",
}


def shortened(value):
    if isinstance(value, str):
        return value[:180]
    if isinstance(value, dict):
        return {key: shortened(item) for key, item in value.items()}
    if isinstance(value, list):
        return [shortened(item) for item in value[:3]]
    return value


for domain, path in SOURCES.items():
    if not path.exists():
        print(json.dumps({"domain": domain, "state": "missing"}))
        continue
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            first = json.loads(next(handle))
        output = {"domain": domain, "keys": list(first), "sample": shortened(first)}
    else:
        source = parquet.ParquetFile(path)
        first = next(source.iter_batches(batch_size=1)).to_pylist()[0]
        output = {
            "domain": domain,
            "rows": source.metadata.num_rows,
            "keys": list(first),
            "sample": shortened(first),
        }
    print(json.dumps(output, ensure_ascii=False))

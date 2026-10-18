#!/root/miniconda3/envs/spec/bin/python
"""Freeze non-overlapping AutoDL teacher shard ownership."""

import hashlib
import json
import os
import pathlib

V2 = pathlib.Path('/root/autodl-tmp/sd-lora-preexp/experiments/v2-routing')
path = V2 / 'data/processed/pilot_manifest.jsonl'
rows = [json.loads(line) for line in path.read_text().splitlines() if line]
requested = [r for r in rows if r['partition'] in ('expert_train', 'adapter_val')]
ownership = {}
for r in requested:
    shard = int(hashlib.sha256(r['record_id'].encode()).hexdigest()[:16], 16) % 2
    assert r['record_id'] not in ownership
    ownership[r['record_id']] = shard
counts = {str(i): sum(index == i for index in ownership.values()) for i in (0,1)}
assert sum(counts.values()) == 1520
manifest = json.loads((V2 / 'data/processed/manifest.json').read_text())
assert manifest['pilot_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
result = {'state': 'frozen', 'manifest_sha256': manifest['pilot_sha256'],
          'algorithm': 'sha256(record_id) first64bits modulo2', 'partition_scope':
          ['expert_train', 'adapter_val'], 'requested': len(ownership),
          'shard_counts': counts, 'owner_set_hashes':
          {str(i): hashlib.sha256('\n'.join(sorted(x for x,v in ownership.items() if v == i)).encode()).hexdigest()
           for i in (0,1)}}
out = V2 / 'status/teacher_sharding.json'
temp = out.with_name(out.name + f'.{os.getpid()}.tmp')
temp.write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
os.replace(temp, out)
print(json.dumps(result, sort_keys=True))

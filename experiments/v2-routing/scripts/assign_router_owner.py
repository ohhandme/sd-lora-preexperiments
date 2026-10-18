#!/root/miniconda3/envs/spec/bin/python
"""Record the explicit v2 router-label ownership transfer before AutoDL work."""

import hashlib
import json
import os
import pathlib
from datetime import datetime, timezone

V2 = pathlib.Path('/root/autodl-tmp/sd-lora-preexp/experiments/v2-routing')
manifest = json.loads((V2/'data/processed/manifest.json').read_text())
path = V2/'status/router_owner.json'
if path.exists():
    old = json.loads(path.read_text())
    assert old['owner'] == 'autodl' and old['pilot_sha256'] == manifest['pilot_sha256']
    print(json.dumps(old, sort_keys=True))
else:
    entries = [json.loads(line) for line in (V2/'data/processed/pilot_manifest.jsonl').read_text().splitlines() if line]
    router = [r for r in entries if r['partition'] in ('router_train','router_val')]
    assert len(router) == 600 and len({r['record_id'] for r in router}) == 600
    ticket = {'state':'assigned','owner':'autodl',
              'at':datetime.now(timezone.utc).isoformat(),
              'pilot_sha256':manifest['pilot_sha256'],
              'partitions':['router_train','router_val'],
              'requested':600,
              'ids_sha256':hashlib.sha256('\n'.join(sorted(r['record_id'] for r in router)).encode()).hexdigest(),
              'previous_4090_output_count':0,
              'reason':'4090 private-network SSH unreachable; separate offload task told not to launch these IDs',
              'no_overlap_with_autodl_train_teacher':True}
    temp = path.with_name(path.name+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(ticket,indent=2,sort_keys=True)+'\n')
    os.replace(temp,path)
    print(json.dumps(ticket, sort_keys=True))

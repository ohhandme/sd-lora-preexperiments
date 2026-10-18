#!/root/miniconda3/envs/spec/bin/python
"""Read-only concise status for the 15-minute v2 autoresearch heartbeat."""

import hashlib
import json
import pathlib
import shutil
import subprocess
from datetime import datetime, timezone

ROOT = pathlib.Path('/root/autodl-tmp/sd-lora-preexp')
V2 = ROOT/'experiments/v2-routing'


def read(path):
    return json.loads(path.read_text()) if path.is_file() else None


def living(pid):
    return bool(pid and pathlib.Path(f'/proc/{pid}/cmdline').is_file())


def main():
    methods = ('code','translation','summary','qa','math','mix16','mix80')
    teacher = [read(V2/f'status/teacher.{i}.json') for i in (0,1)]
    router = [read(V2/f'status/router_teacher.{i}.json') for i in (0,1)]
    selected = {m:read(V2/'adapters/0.6B'/m/'seed42/selected.json') is not None
                for m in methods}
    workers = {m:read(V2/'status'/f'train.{m}.json') for m in methods}
    quality = read(V2/'status/quality.json')
    routing = read(V2/'status/routing.json')
    online = {m:read(V2/'status'/f'online.{m}.json') for m in
              ('AR8','BASE','mix16','mix80','DOMAIN','ROUTE-TASK','ROUTE-ACC')}
    coordinator = read(V2/'status/coordinator.json')
    gpu = subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total,power.draw',
                          '--format=csv,noheader'],capture_output=True,text=True)
    disk = shutil.disk_usage(ROOT)
    result = {'at':datetime.now(timezone.utc).isoformat(),
              'v1_program_sha256':hashlib.sha256((ROOT/'program.md').read_bytes()).hexdigest(),
              'v2_program_sha256':hashlib.sha256((V2/'program.md').read_bytes()).hexdigest(),
              'teacher':teacher,'router_teacher':router,
              'router_owner':read(V2/'status/router_owner.json'),
              'coordinator':coordinator,
              'coordinator_pid_live':living(coordinator.get('pid') if coordinator else None),
              'selected_count':sum(selected.values()),'selected':selected,
              'train_workers':{m:v for m,v in workers.items() if v is not None},
              'quality':quality,'routing':routing,'online':
              {m:v for m,v in online.items() if v is not None},
              'final_report':read(V2/'results/final_report.json'),
              'gpu':gpu.stdout.strip(),'gpu_query_returncode':gpu.returncode,
              'disk_free_bytes':disk.free}
    print(json.dumps(result,sort_keys=True))


if __name__ == '__main__':
    main()

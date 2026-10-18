#!/root/miniconda3/envs/spec/bin/python
"""Resume-safe v2 pilot pipeline owner; stops on evidence-bearing failures."""

import json
import os
import pathlib
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = pathlib.Path('/root/autodl-tmp/sd-lora-preexp')
V2 = ROOT / 'experiments/v2-routing'
PYTHON = '/root/miniconda3/envs/spec/bin/python'
METHODS = ('code', 'translation', 'summary', 'qa', 'math', 'mix16', 'mix80')
STATUS = V2 / 'status/coordinator.json'
CHILD = None
STOP = False


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic(value):
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    temp = STATUS.with_name(STATUS.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    os.replace(temp, STATUS)


def request_stop(*_):
    global STOP
    STOP = True


def state(value, **details):
    atomic({'at': now(), 'state': value, 'pid': os.getpid(),
            'interpreter': sys.executable, **details})


def teacher_state():
    shard_paths = [V2 / f'status/teacher.{i}.json' for i in (0, 1)]
    if any(p.is_file() for p in shard_paths):
        shards = [json.loads(p.read_text()) if p.is_file() else {'state': 'missing'}
                  for p in shard_paths]
        if any(s['state'] == 'failed' for s in shards):
            return {'state': 'failed', 'shards': shards}
        return {'state': 'complete' if all(s['state'] == 'complete' for s in shards)
                else 'running', 'completed': sum(s.get('completed', 0) for s in shards),
                'requested': sum(s.get('requested', 0) for s in shards), 'shards': shards}
    p = V2 / 'status/teacher.json'
    return json.loads(p.read_text()) if p.is_file() else {'state': 'missing'}


def router_teacher_state():
    paths = [V2 / f'status/router_teacher.{i}.json' for i in (0,1)]
    shards = [json.loads(p.read_text()) if p.is_file() else {'state':'missing'}
              for p in paths]
    if any(s['state'] == 'failed' for s in shards):
        return {'state':'failed','shards':shards}
    return {'state':'complete' if all(s['state']=='complete' for s in shards)
            else 'running', 'completed':sum(s.get('completed',0) for s in shards),
            'requested':sum(s.get('requested',0) for s in shards), 'shards':shards}


def wait_exited(shards):
    for _ in range(24):
        living = []
        for shard in shards:
            pid = shard.get('pid')
            if pid:
                try:
                    os.kill(pid, 0)
                    living.append(pid)
                except ProcessLookupError:
                    pass
        if not living:
            return
        state('waiting_teacher_exit', living_pids=living)
        time.sleep(5)
    raise RuntimeError(f'Teacher processes still resident after completion: {living}')


def main():
    global CHILD
    assert sys.executable == PYTHON
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGUSR1, request_stop)
    if not (V2 / 'status/pilot_duplicates.json').is_file():
        raise RuntimeError('Pilot near-duplicate audit is missing')
    if json.loads((V2 / 'status/pilot_duplicates.json').read_text())['state'] != 'passed':
        raise RuntimeError('Pilot near-duplicate audit failed')
    gate = json.loads((V2 / 'status/teacher_gate.json').read_text())
    if gate['state'] != 'passed' or not gate['full_token_eos_truncation_equal']:
        raise RuntimeError('Teacher batch correctness gate failed')
    while not STOP:
        teacher = teacher_state()
        if teacher['state'] == 'failed':
            state('failed', stage='teacher', teacher=teacher)
            return 1
        if teacher['state'] == 'complete':
            assert teacher['completed'] == teacher['requested'] == 1520
            break
        state('waiting_teacher', completed=teacher.get('completed', 0), requested=1520)
        time.sleep(30)
    if STOP:
        state('paused', stage='teacher')
        return 0
    wait_exited(teacher.get('shards',[]))
    ticket_path = V2 / 'status/router_owner.json'
    if ticket_path.is_file():
        ticket = json.loads(ticket_path.read_text())
        assert ticket['state'] == 'assigned' and ticket['owner'] == 'autodl'
        assert ticket['requested'] == 600
        launch = subprocess.run(['bash',str(V2/'scripts/launch_router_teacher.sh')],
                                capture_output=True,text=True)
        if launch.returncode:
            state('failed',stage='router_teacher_launch',returncode=launch.returncode,
                  stdout=launch.stdout,stderr=launch.stderr)
            return launch.returncode
        while not STOP:
            router = router_teacher_state()
            if router['state'] == 'failed':
                state('failed',stage='router_teacher',router=router)
                return 1
            if router['state'] == 'complete':
                assert router['completed'] == router['requested'] == 600
                wait_exited(router['shards'])
                break
            state('waiting_router_teacher',completed=router['completed'],requested=600)
            time.sleep(30)
        if STOP:
            state('paused',stage='router_teacher')
            return 0
    def live_worker(method):
        pidfile = V2/'status'/f'worker.{method}.pid'
        if not pidfile.is_file():
            return None
        try:
            pid = int(pidfile.read_text().strip())
            command = pathlib.Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
        except (ValueError, FileNotFoundError, ProcessLookupError):
            return None
        return pid if 'train.py' in command and f'--method {method}' in command else None

    existing = {method: live_worker(method) for method in METHODS}
    while any(existing.values()) and not STOP:
        state('waiting_existing_workers',stage='training',pids=existing)
        time.sleep(10)
        existing = {method: live_worker(method) for method in METHODS}
    for method in METHODS:
        pidfile = V2/'status'/f'worker.{method}.pid'
        if pidfile.is_file() and not live_worker(method):
            pidfile.unlink()
    pending = [method for method in METHODS
               if not (V2/'adapters/0.6B'/method/'seed42/selected.json').is_file()]
    running = {}
    failures = []
    while pending or running:
        while pending and len(running) < 2 and not STOP and not failures:
            method = pending.pop(0)
            log = V2/'logs'/f'worker.{method}.log'
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open('a') as out:
                child = subprocess.Popen(
                    ['flock','-s','-w','120',str(ROOT/'manifests/gpu_pipeline.lock'),
                     PYTHON,str(V2/'scripts/train.py'),'--method',method],
                    stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,
                    start_new_session=True)
            (V2/'status'/f'worker.{method}.pid').write_text(str(child.pid)+'\n')
            running[method] = child
        finished = []
        for method, child in running.items():
            code = child.poll()
            if code is None:
                continue
            selected = V2/'adapters/0.6B'/method/'seed42/selected.json'
            if code != 0 or not selected.is_file():
                failures.append({'method':method,'returncode':code,
                                 'selected_exists':selected.is_file(),
                                 'log':str(V2/'logs'/f'worker.{method}.log')})
            else:
                result = json.loads(selected.read_text())
                assert result['selected_epoch'] in (1,2)
            finished.append(method)
        for method in finished:
            del running[method]
            (V2/'status'/f'worker.{method}.pid').unlink(missing_ok=True)
        state('running' if not failures and not STOP else 'draining',stage='training',
              running_methods=list(running),pending_methods=pending,
              selected_count=sum((V2/'adapters/0.6B'/m/'seed42/selected.json').is_file()
                                 for m in METHODS),total_methods=len(METHODS),failures=failures)
        if running:
            time.sleep(10)
        elif failures or STOP:
            break
    if failures:
        state('failed',stage='training',failures=failures)
        return 1
    if STOP:
        state('paused',stage='training',pending_methods=pending)
        return 0
    quality_summary = V2 / 'results/quality/summary.json'
    if not quality_summary.is_file():
        log = V2 / 'logs/worker.quality.log'
        state('running', stage='quality', selected_count=len(METHODS), total_methods=len(METHODS))
        with log.open('a') as out:
            CHILD = subprocess.Popen(
                ['flock', '-w', '120', str(ROOT / 'manifests/gpu_pipeline.lock'),
                 PYTHON, str(V2 / 'scripts/quality.py')],
                stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            returncode = CHILD.wait()
        CHILD = None
        if returncode != 0:
            state('failed', stage='quality', returncode=returncode, log=str(log))
            return returncode
    quality = json.loads(quality_summary.read_text())
    if quality['state'] != 'complete' or quality['completed_entries'] != 1920:
        state('failed', stage='quality', reason='Incomplete quality matrix', quality=quality)
        return 3
    state('quality_complete', selected_count=len(METHODS), total_methods=len(METHODS),
          quality_headroom_only=quality['quality_headroom_only'],
          next_stage='route_labels_and_cost_gate')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

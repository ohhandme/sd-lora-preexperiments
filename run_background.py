"""Persistent supervisor. Waits for existing installer, then P0, then gated P1/P2."""
import argparse,subprocess,time,traceback
from spec_research.common import *

def child(args,log,pidfile):
    pathlib.Path(log).parent.mkdir(parents=True,exist_ok=True)
    with open(log,'a',buffering=1) as f:
        p=subprocess.Popen(args,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        pathlib.Path(pidfile).write_text(str(p.pid)+'\n')
        rc=p.wait()
    assert rc==0,f'Child exit {rc}: {args}; log={log}'
if __name__=='__main__':
    assert_spec();p=argparse.ArgumentParser();p.add_argument('--p0-run-id',required=True);p.add_argument('--run-id',required=True);a=p.parse_args()
    import fcntl
    lock=(ROOT/'manifests/background-supervisor.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        while True:
            setup=read(ROOT/'manifests/environment/spec_setup.json')
            if setup['state']=='complete':break
            assert setup['state']=='running',f'Environment installer failed: {setup}'
            os.kill(setup['pid'],0)
            status(state='running',phase='waiting_for_spec_installer',training_started=False,next_action='Wait for existing authorized installer; do not race another install')
            time.sleep(10)
        child([SPEC+'/bin/python','-u','smoke_test.py','--run-id',a.p0_run_id],ROOT/'logs/smoke/p0-worker.log',ROOT/'manifests/p0-worker.pid')
        child([SPEC+'/bin/python','validate_gate.py'],ROOT/'logs/smoke/independent_guard.log',ROOT/'manifests/guard-worker.pid')
        child([SPEC+'/bin/python','-u','run_seed42_pipeline.py','--run-id',a.run_id],ROOT/'logs/train/seed42-worker.log',ROOT/'manifests/seed42-worker.pid')
    except BaseException as e:
        old=read(ROOT/'manifests/execution_status.json')
        # Preserve the more precise P0 failure emitted by the child.
        if old.get('state')!='blocked':status(state='blocked',phase='background_supervisor_failed',training_started=old.get('training_started',False),latest_error=repr(e),next_action='Inspect installer/smoke/train evidence; formal launch remains fail-closed')
        append(ROOT/'logs/smoke/supervisor_failures.jsonl',{'at':now(),'error':repr(e),'traceback':traceback.format_exc()})
        raise

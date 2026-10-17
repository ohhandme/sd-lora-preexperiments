"""Change supervisor only at a clean post-training boundary; never kill training."""
import pathlib,sys,time,os,signal,subprocess,fcntl
ROOT=pathlib.Path('/root/autodl-tmp/sd-lora-preexp');sys.path.insert(0,str(ROOT))
from spec_research.common import *
from spec_research.gate import validate_gate
assert_spec()

def alive(pid):
    try:return pathlib.Path(f'/proc/{pid}/stat').read_text().split()[2]!='Z'
    except FileNotFoundError:return False

with (ROOT/'manifests/scope_transition.lock').open('a') as owner:
    fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while True:
        cfg=read(ROOT/'manifests/continuation_config.json');assert cfg['online_scope']=='preliminary24'
        s=read(ROOT/'manifests/continuation_status.json')
        if s['state'] in ('needs_repair','interrupted','complete'):
            atomic(ROOT/'manifests/scope_transition_status.json',dict(state='needs_review',at=now(),reason=s['state']));break
        assert cfg['evaluation_ready'] is False,'Old supervisor must remain behind the post-training boundary'
        selected=all((ROOT/f'adapters/{z}/{m}/seed42/overnight-20261017-seed42/selected.json').exists() for z in ('0.6B','4B') for m in (*DOMAINS,'mix16','mix48'))
        if not (selected and not s['workers'] and cfg.get('scope_transition_ready')):
            atomic(ROOT/'manifests/scope_transition_status.json',dict(state='waiting_for_training_and_selection',at=now(),selected_all=selected,workers=list(s['workers'])))
            time.sleep(20);continue
        validate_gate();check=read(ROOT/'results/continuation-20261017/evaluation_implementation_checks.json')
        assert check['state']=='passed'
        for name,sha in check['helpers'].items():assert digest(ROOT/'scripts/continuation'/name)==sha
        old=s['coordinator_pid'];cmd=pathlib.Path(f'/proc/{old}/cmdline').read_bytes().decode()
        assert str(ROOT/'scripts/continuation/continuation_coordinator.py') in cmd
        atomic(ROOT/'results/continuation-20261017/scope_transition_old_status.json',s)
        os.kill(old,signal.SIGTERM)
        limit=time.monotonic()+45
        while alive(old) and time.monotonic()<limit:time.sleep(.5)
        assert not alive(old),'Old coordinator did not exit cleanly; do not duplicate it'
        cfg['evaluation_ready']=True;cfg['scope_transition_completed_at']=now();cfg.pop('evaluation_hold_reason',None)
        atomic(ROOT/'manifests/continuation_config.json',cfg)
        log=ROOT/'logs/continuation-20261017/coordinator.preliminary24.log'
        with log.open('a') as stream:
            child=subprocess.Popen([SPEC+'/bin/python','-u',str(ROOT/'scripts/continuation/continuation_coordinator.py')],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stream,stderr=stream,start_new_session=True)
        atomic(ROOT/'manifests/scope_transition_status.json',dict(state='started_preliminary_supervisor',at=now(),old_pid=old,new_pid=child.pid))
        break

"""Finite continuation queue, checkpoint-safe training and exclusive evaluation."""
import pathlib,sys,os,time,json,signal,subprocess,fcntl,traceback,shutil
ROOT=pathlib.Path('/root/autodl-tmp/sd-lora-preexp');sys.path.insert(0,str(ROOT))
from spec_research.common import *
from spec_research.gate import validate_gate
BASE=ROOT/'results/continuation-20261017';STATE=ROOT/'manifests/continuation_status.json'
CONFIG=ROOT/'manifests/continuation_config.json';SCRIPT=ROOT/'scripts/continuation'
assert_spec();BASE.mkdir(parents=True,exist_ok=True)
(ROOT/'logs/continuation-20261017').mkdir(parents=True,exist_ok=True)
cfg=read(CONFIG);workers={};completed=set();failed={};attempts={};stop_requested=False

def alive(pid):
    try:return pathlib.Path(f'/proc/{pid}/stat').read_text().split()[2]!='Z'
    except FileNotFoundError:return False
def stop(pid):
    try:
        cmd=pathlib.Path(f'/proc/{pid}/cmdline').read_bytes().decode(errors='replace')
        if str(ROOT) not in cmd or not alive(pid):return
        os.killpg(pid,signal.SIGTERM)
        limit=time.monotonic()+15
        while alive(pid) and time.monotonic()<limit:time.sleep(.2)
        if alive(pid):os.killpg(pid,signal.SIGKILL)
    except ProcessLookupError:pass
def resources():
    p=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True)
    assert p.returncode==0,p.stderr
    u,m,t=[int(x.strip()) for x in p.stdout.strip().split(',')]
    return dict(utilization=u,used_mib=m,total_mib=t)

if '--watchdog' in sys.argv:
    while True:
        time.sleep(30)
        if not STATE.exists():continue
        s=read(STATE)
        if s.get('state') in ('complete','needs_repair','interrupted'):break
        reason=None
        if not alive(s['coordinator_pid']):reason='Coordinator exited; clean up only recorded owned tasks'
        for name,w in s.get('workers',{}).items():
            progress=max([pathlib.Path(p).stat().st_mtime for p in w['progress_paths'] if pathlib.Path(p).exists()]+[w['started_unix']])
            if time.time()-progress>1200:reason=f'No worker progress for20min: {name}'
        if reason:
            atomic(ROOT/'manifests/continuation_stop.json',dict(at=now(),reason=reason))
            for w in s.get('workers',{}).values():stop(w['pid'])
            s.update(state='needs_repair',reason=reason,workers={},finished_at=now());atomic(STATE,s);break
    sys.exit(0)

def update(phase,state='running',**extra):
    s=dict(at=now(),state=state,phase=phase,coordinator_pid=os.getpid(),environment=SPEC,
      completed_jobs=sorted(completed),failed_jobs=failed,attempts=attempts,program_sha256=digest(ROOT/'program.md'),
      workers={n:dict(pid=w['proc'].pid,kind=w['kind'],log=str(w['log']),started_unix=w['started'],progress_paths=w['progress']) for n,w in workers.items()})
    s.update(extra);atomic(STATE,s)
def request_stop(signum,frame):
    global stop_requested
    stop_requested=True
signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)

def start(name,args,kind,progress):
    assert attempts.get(name,0)<3,'Retry cap reached; diagnose before retry'
    attempts[name]=attempts.get(name,0)+1
    log=ROOT/'logs/continuation-20261017'/f'{name}.attempt{attempts[name]}.log';log.parent.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,SD_RESULTS_REL='results/continuation-20261017')
    cmd=[SPEC+'/bin/python','-u',str(args[0]),*args[1:],'--managed-lock-fd',str(lock_fd)]
    with log.open('a') as stream:proc=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stream,stderr=stream,start_new_session=True,pass_fds=(lock_fd,),env=env)
    workers[name]=dict(proc=proc,kind=kind,args=args,log=log,started=time.time(),progress=[str(log),*[str(p) for p in progress]])
    atomic(BASE/'job_runtime'/f'{name}.attempt{attempts[name]}.json',dict(at=now(),command=cmd,source_hashes=sources(),helpers={str(p.relative_to(ROOT)):digest(p) for p in SCRIPT.glob('*.py')}))
    append(BASE/'events.jsonl',dict(at=now(),event='started',name=name,pid=proc.pid,attempt=attempts[name]));update('jobs_running')

def verify_training(scale,method):
    p=ROOT/f'adapters/{scale}/{method}/seed42/overnight-20261017-seed42';c=read(p/'complete.json');sel=read(p/'selected.json');n=6000 if method.startswith('mix') else 2000
    assert len(c['exposure_counts'])==n and set(c['exposure_counts'].values())=={2}
    assert c['steps']==(750 if n==6000 else 250) and sel['provenance']['source_hashes']==sources()
    return dict(state='complete',scale=scale,method=method,adapter=str(p),selected=sel,at=now())

def verify_evaluation(scale,kind):
    if kind.startswith('quality'):
        path=BASE/'quality'/scale/'matrix.jsonl';keys={(r['id'],r['method'],r['L']) for r in rows(path)}
        lengths=(8,) if kind=='quality8' else (4,12)
        for L in lengths:assert sum(k[2]==L for k in keys)==3600
        assert read(path.parent/'quality_summary.json')['completed_matrix_requests']==600
    elif kind=='router':
        path=BASE/'routing'/scale/'labels.jsonl';assert len({r['id'] for r in rows(path)})==1500
        v=read(path.parent/'router_validation.json');assert v['state']=='complete' and v['train_requests']+v['val_requests']+v['excluded_no_public_window']==1500
        assert (ROOT/f'routers/acceptance/{scale}/seed42/overnight-20261017-seed42/router.pkl').exists()
    elif kind=='preliminary':
        path=BASE/'preliminary_online'/scale/'requests.jsonl';v=read(path.parent/'summary.json');subset=read(path.parent/'subset_manifest.json')
        assert v['state']=='complete' and v['SD_comparisons']==168 and v['independently_measured_AR_requests']==24
        assert subset['requests']==24 and subset['per_domain']==8 and subset['timing_repeats']==1 and subset['L']==8
        keys=set()
        for r in rows(path):
            assert r['token_match'] and r['eos_match'] and r['truncation_match'] and r['ar']['tokens']==r['sd']['tokens']
            keys.add((r['repeat'],r['method'],r['L'],r['id']))
        expected={(0,m,8,i) for m in ('BASE','mix16','mix48','DOMAIN','FIXEDBEST','ROUTE-TASK','ROUTE-ACC') for i in subset['ids']}
        assert keys==expected
        profiles=list((path.parent/'configurations').glob('*/component_profile.json'))
        assert len(profiles)==7 and all(read(p)['sample_count']==3 for p in profiles)
    elif kind.startswith('online'):
        path=BASE/'online'/scale/'requests.jsonl';keys=set()
        for r in rows(path):
            assert r['token_match'] and r['ar']['tokens']==r['sd']['tokens'];keys.add((r['repeat'],r['method'],r['L'],r['id']))
        lengths=(8,) if kind=='online8' else (4,12)
        for L in lengths:assert sum(k[2]==L for k in keys)==600*3*7
    else:raise AssertionError(kind)
    atomic(BASE/'verified'/f'{kind}.{scale}.json',dict(at=now(),state='complete',scale=scale,kind=kind,source_hashes=sources()))

def finish_preliminary():
    groups={};stages={}
    for s in ('0.6B','4B'):
        for m in (*DOMAINS,'mix16','mix48'):
            groups[s+'.'+m]=verify_training(s,m)
        for k in ('quality8','router','preliminary'):verify_evaluation(s,k)
        stages[s]=dict(quality=read(BASE/'quality'/s/'quality_summary.json'),
                       routing=read(BASE/'routing'/s/'router_validation.json'),
                       online=read(BASE/'preliminary_online'/s/'summary.json'),
                       reused_deployment_report=str(ROOT/'results/overnight-20261017/deployment'/s/'report.json'))
    a=read(BASE/'preliminary_online/0.6B/subset_manifest.json');b=read(BASE/'preliminary_online/4B/subset_manifest.json');assert a==b
    atomic(BASE/'final_report.json',dict(state='complete',scope='User-authorized preliminary24 stage only',finished_at=now(),
        program_unchanged=True,program_sha256=digest(ROOT/'program.md'),selected_groups=groups,stages=stages,
        counts=dict(seed42_selections=10,quality_L8_matrix_entries=7200,router_labeled_requests=3000,online_SD_comparisons=336,independently_measured_AR_requests=48),
        deferred=['600-request full online matrix','L4/L12 scans and L* selection','timing repeats2/3','seed43/44','locked final ID and external tests','real WAN/SLO/async second research point'],
        limitations=['24 requests eight per domain and one timing repeat: preliminary trends only, no robust final improvement claim.',
                    'Historical merged-copy numerical differences preserved; local GPU latency is not network latency.'],
        results_directory=str(BASE)))
    atomic(pathlib.Path(cfg['autoresearch_directory'])/'handoff.json',dict(state='complete_preliminary24',report=str(BASE/'final_report.json'),
        next_action='Heartbeat confirms no remaining GPU tasks, reports true preliminary findings, pauses monitor; full protocol remains deferred'))

def reap():
    for name,w in list(workers.items()):
        rc=w['proc'].poll()
        if rc is None:continue
        del workers[name]
        if rc==0:
            parts=name.split('.')
            if w['kind']=='train':
                scale=w['args'][w['args'].index('--scale')+1];method=w['args'][w['args'].index('--method')+1];verify_training(scale,method)
            else:
                scale=w['args'][w['args'].index('--scale')+1];verify_evaluation(scale,name.rsplit('.',1)[0] if scale=='4B' else name[:-5])
            completed.add(name);append(BASE/'events.jsonl',dict(at=now(),event='verified_complete',name=name))
        else:
            tail=w['log'].read_text()[-6000:];failed[name]=dict(at=now(),exit_code=rc,log=str(w['log']),tail=tail)
            append(BASE/'events.jsonl',dict(at=now(),event='failed',name=name,exit_code=rc))
            if ('OutOfMemory' in tail or 'out of memory' in tail) and attempts[name]<2:
                cfg['max_parallel_training']=1;failed.pop(name);atomic(CONFIG,cfg)
            else:raise RuntimeError(f'{name} failed; evidence saved, no blind replay')

watchdog=None
try:
    owner=(ROOT/'manifests/background-supervisor.lock').open('a');fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
    context=gpu_lock();lock_fd=context.__enter__();validate_gate()
    assert digest(ROOT/'program.md')==cfg['program_sha256']
    train_jobs=[('4B','mix16'),('0.6B','mix48'),('4B','mix48')]
    all_groups=[(s,m) for s in ('0.6B','4B') for m in (*DOMAINS,'mix16','mix48')]
    previous=read(STATE) if STATE.exists() else {};attempts.update(previous.get('attempts',{}));completed.update(previous.get('completed_jobs',[]))
    for scale,method in all_groups:
        p=ROOT/f'adapters/{scale}/{method}/seed42/overnight-20261017-seed42'
        if (p/'selected.json').exists():
            record=verify_training(scale,method);atomic(BASE/'jobs'/f'{scale}.{method}.json',record);completed.add(f'train.{scale}.{method}')
    stop_file=ROOT/'manifests/continuation_stop.json'
    if stop_file.exists():raise RuntimeError('Unresolved watchdog stop marker; archive after diagnosis before restart')
    update('resuming_training')
    with (ROOT/'logs/continuation-20261017/watchdog.log').open('a') as stream:watchdog=subprocess.Popen([SPEC+'/bin/python','-u',__file__,'--watchdog'],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stream,stderr=stream,start_new_session=True)
    kinds=('quality8','router','preliminary') if cfg.get('online_scope')=='preliminary24' else ('quality8','router','online8','quality_scan','online_scan')
    eval_jobs=[(s,k) for k in kinds for s in ('0.6B','4B')]
    next_check=0
    while True:
        if stop_requested:raise InterruptedError('Coordinator interrupted; checkpoints retained')
        if stop_file.exists():raise RuntimeError(read(stop_file)['reason'])
        reap();gpu=resources();cfg=read(CONFIG)
        if time.time()>=next_check:
            next_check=time.time()+900;append(BASE/'checks_15min.jsonl',dict(at=now(),gpu=gpu,completed=sorted(completed),workers={k:v['proc'].pid for k,v in workers.items()},failed=failed))
        if all(f'train.{s}.{m}' in completed for s,m in all_groups):
            if not workers:
                if not cfg.get('evaluation_ready',False):
                    update('evaluation_code_verification_pending',gpu=gpu);time.sleep(10);continue
                pending=next(((s,k) for s,k in eval_jobs if f'{k}.{s}' not in completed),None)
                if pending is None:
                    if cfg.get('online_scope')=='preliminary24':finish_preliminary()
                    status(state='complete',phase='preliminary24_complete' if cfg.get('online_scope')=='preliminary24' else 'seed42_validation_complete',supervisor_running=False,gpu_worker_pids=[],latest_error=None)
                    update('all_seed42_validation_jobs_verified',state='complete',gpu=gpu);break
                scale,kind=pending;action='preliminary' if kind=='preliminary' else ('router' if kind=='router' else ('quality' if kind.startswith('quality') else 'online'))
                args=[ROOT/'scripts/continuation/continuation_evaluate.py',action,'--scale',scale]
                if action!='router':args+=['--lengths','8' if kind.endswith('8') or action=='preliminary' else '4,12']
                folder=BASE/({'router':'routing','quality':'quality','online':'online','preliminary':'preliminary_online'}[action])/scale
                progress=[folder/p for p in ('matrix.jsonl','labels.jsonl','requests.jsonl','router_validation.json')]
                start(f'{kind}.{scale}',args,'evaluation',progress)
        else:
            for scale,method in train_jobs:
                name=f'train.{scale}.{method}'
                if name in completed or name in workers:continue
                if len(workers)>=cfg['max_parallel_training']:break
                required=34816 if scale=='4B' else 10240
                reserve=sum(34816 if w['args'][w['args'].index('--scale')+1]=='4B' else 10240 for w in workers.values())
                if reserve+required>gpu['total_mib']-6144 or gpu['total_mib']-gpu['used_mib']<required:continue
                folder=ROOT/f'adapters/{scale}/{method}/seed42/overnight-20261017-seed42'
                progress=[folder/'resume.pt',ROOT/f'results/validation/overnight-20261017-seed42.{scale}.{method}.selection.jsonl',BASE/f'selection_progress/{scale}.{method}.json']
                start(name,[SCRIPT/'continuation_worker.py','--scale',scale,'--method',method],'train',progress)
        update('training' if any(w['kind']=='train' for w in workers.values()) else 'evaluation',gpu=gpu)
        status(state='running',phase='continuation_'+('training' if any(w['kind']=='train' for w in workers.values()) else 'evaluation'),supervisor_pid=os.getpid(),supervisor_running=True,gpu_worker_pids=[w['proc'].pid for w in workers.values()],latest_error=None,next_action='Complete finite seed42 queue; all timing evaluations exclusive')
        time.sleep(10)
except BaseException as e:
    for w in workers.values():stop(w['proc'].pid)
    workers.clear();update('requires_diagnosis',state='interrupted' if isinstance(e,InterruptedError) else 'needs_repair',error=repr(e),traceback=traceback.format_exc())
    status(state='needs_repair',phase='continuation_requires_diagnosis',supervisor_running=False,gpu_worker_pids=[],latest_error=repr(e))
    append(BASE/'failures.jsonl',dict(at=now(),error=repr(e),traceback=traceback.format_exc()))
finally:
    if watchdog is not None and watchdog.poll() is None:watchdog.terminate()
    if 'context' in globals():context.__exit__(None,None,None)
    if 'owner' in globals():owner.close()

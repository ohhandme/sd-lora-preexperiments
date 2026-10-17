import argparse,subprocess,traceback
from spec_research.common import *
from spec_research.gate import validate_gate
from spec_research.data import partition,validate_data
from spec_research.teacher import generate,validate_teacher
from spec_research.training import train
from spec_research.evaluation import select_checkpoint

def pipeline(run_id,lock_fd):
    validate_gate();validate_data();runfile=ROOT/'manifests'/f'pipeline.{run_id}.json'
    spec={'run_id':run_id,'source_hashes':sources(),'data_sha':digest(ROOT/'manifests/data.json'),'gate_sha':digest(ROOT/'manifests/p0_gate.json'),'prefix':sys.prefix,'jobs':[[s,m] for s in ('0.6B','4B') for m in (*DOMAINS,'mix16','mix48')],'provenance':provenance()}
    if runfile.exists():
        old=read(runfile);assert all(old[k]==spec[k] for k in ['source_hashes','data_sha','gate_sha','prefix','jobs'])
    else:atomic(runfile,spec)
    old_status=read(ROOT/'manifests/execution_status.json')
    status(phase='teacher_running',state='running',run_id=run_id,pipeline_pid=os.getpid(),training_started=old_status.get('training_started',False),latest_error=None,next_action='Complete and validate all 6600 teacher trajectories before formal training')
    rr=[r for d in DOMAINS for p in ('expert_train','adapter_val') for r in partition(d,p)];assert len(rr)==6600
    def gpu_child(args,log):
        with open(log,'a') as stream:
            proc=subprocess.Popen([SPEC+'/bin/python','-u',*args,'--managed-lock-fd',str(lock_fd)],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,pass_fds=(lock_fd,))
            pathlib.Path(ROOT/'manifests/current-gpu-worker.pid').write_text(str(proc.pid)+'\n')
            rc=proc.wait()
        assert rc==0,f'GPU child failed exit {rc}; see {log}'
    gpu_child(['generate_teacher.py','--run-id',run_id],ROOT/'logs/teacher'/f'{run_id}.worker.log')
    tt=validate_teacher(rr);assert len(tt)==6600
    atomic(ROOT/'manifests'/f'teacher.{run_id}.json',{'count':6600,'record_sha':objhash([r['record_sha'] for r in tt]),'config_sha':tt[0]['config_sha'],'provenance':provenance()})
    jobs=[]
    for scale,method in spec['jobs']:
        validate_gate();rr=[r for r in tt if r['partition']=='expert_train' and (method.startswith('mix') or r['domain']==method)]
        out=ROOT/'adapters'/scale/method/'seed42'/run_id
        gpu_child(['train_lora.py','--scale',scale,'--method',method,'--run-id',run_id,'--select-checkpoint'],ROOT/'logs/train'/f'{run_id}.{scale}.{method}.worker.log')
        tr=read(out/'complete.json');assert tr['state']=='complete'
        selected=read(out/'selected.json');jobs.append({'scale':scale,'method':method,'complete':str(out/'complete.json'),'selected':selected})
        atomic(ROOT/'results/validation'/f'{run_id}.jobs.json',jobs)
    status(phase='seed42_complete',state='complete',training_started=True,current_job=None,counts={'completed_jobs':10,'teacher_records':6600},next_action='Review validation checkpoint evidence; P3/final tests remain gated')
    atomic(ROOT/'autoresearch/loop-261016-2120/handoff.json',{'status':'seed42_complete','macro_speedup_geo':None,'metric_status':'NOT_RUN','run_id':run_id,'jobs':jobs,'deferred':'Disk-heavy full model exports deferred; use in-memory merge deployment tests later. No tests unlocked.'})
if __name__=='__main__':
    assert_spec();p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);a=p.parse_args()
    try:
        with gpu_lock() as lock_fd:pipeline(a.run_id,lock_fd)
    except BaseException as e:
        old=read(ROOT/'manifests/execution_status.json')
        status(state='blocked',phase='pipeline_failed',training_started=old.get('training_started',False),latest_error=repr(e),next_action='Inspect logs and resume the same immutable run after diagnosis')
        append(ROOT/'logs/train'/f'{a.run_id}.failures.jsonl',{'at':now(),'error':repr(e),'traceback':traceback.format_exc()});raise

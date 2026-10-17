import argparse
from spec_research.common import *
from spec_research.data import partition
from spec_research.teacher import validate_teacher
from spec_research.training import train
if __name__=='__main__':
    assert_spec();p=argparse.ArgumentParser();p.add_argument('--scale',choices=['0.6B','4B'],required=True);p.add_argument('--method',choices=[*DOMAINS,'mix16','mix48'],required=True);p.add_argument('--run-id',required=True);p.add_argument('--managed-lock-fd',type=int);p.add_argument('--select-checkpoint',action='store_true');a=p.parse_args()
    from spec_research.gate import validate_gate
    validate_gate();validate_teacher([r for d in DOMAINS for part in ('expert_train','adapter_val') for r in partition(d,part)])
    domains=DOMAINS if a.method.startswith('mix') else [a.method]
    rr=[r for d in domains for r in partition(d,'expert_train')];tt=validate_teacher(rr)
    out=ROOT/'adapters'/a.scale/a.method/'seed42'/a.run_id
    with gpu_lock(a.managed_lock_fd):
        train(a.scale,a.method,tt,out,ROOT/'logs/train'/f'{a.run_id}.{a.scale}.{a.method}.jsonl')
        if a.select_checkpoint:
            from spec_research.evaluation import select_checkpoint
            status(phase='validation_checkpoint_selection',state='running',current_job=f'{a.scale}/{a.method}')
            select_checkpoint(a.scale,a.method,out)

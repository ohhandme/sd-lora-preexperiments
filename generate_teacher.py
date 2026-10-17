import argparse
from spec_research.common import *
from spec_research.data import partition,smoke_records,validate_data
from spec_research.teacher import generate
if __name__=='__main__':
    assert_spec();p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--run-id',default='standalone');p.add_argument('--managed-lock-fd',type=int);a=p.parse_args();validate_data()
    if not a.smoke:
        from spec_research.gate import validate_gate
        validate_gate()
    rr=smoke_records() if a.smoke else [r for d in DOMAINS for part in ('expert_train','adapter_val') for r in partition(d,part)]
    with gpu_lock(a.managed_lock_fd):generate(rr,ROOT/('logs/smoke' if a.smoke else 'logs/teacher')/(a.run_id+'.jsonl'),'smoke_teacher_running' if a.smoke else 'teacher_running')

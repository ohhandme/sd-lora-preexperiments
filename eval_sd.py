import argparse,math
from spec_research.common import *
from spec_research.data import smoke_records
from spec_research.evaluation import evaluate_requests
if __name__=='__main__':
    assert_spec();p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--metric-only');p.add_argument('--check-correctness',action='store_true');p.add_argument('--results');a=p.parse_args();cfg=read(a.config)
    if a.check_correctness:
        from spec_research.gate import validate_gate
        validate_gate();print('P0_GUARD_PASSED')
    elif a.metric_only:
        assert a.metric_only=='macro_speedup_geo'
        assert a.results,'NOT_RUN: paired adapter_val timing results are required; no fabricated baseline'
        rr=list(rows(a.results));assert rr and all(r['token_match'] and r['partition']=='adapter_val' for r in rr)
        ratios=[]
        for d in DOMAINS:
            dd=[r for r in rr if r['domain']==d];assert dd
            ratios.append(sum(r['ar']['latency_ms'] for r in dd)/sum(r['sd']['latency_ms'] for r in dd))
        value=math.exp(sum(math.log(x) for x in ratios)/3);assert math.isfinite(value);print(value)
    else:raise SystemExit('Specify --check-correctness or --metric-only with real paired results')

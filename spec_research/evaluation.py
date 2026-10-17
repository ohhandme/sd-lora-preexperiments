import gc,statistics,time
import torch
from peft import PeftModel
from .common import *
from .decoding import ar,speculative,compare,public_prefix
from .data import partition
from .teacher import validate_teacher

def eos_ids(model):
    eos=model.generation_config.eos_token_id;return set([eos] if isinstance(eos,int) else eos)
def evaluate_requests(scale,records,adapter,out,max_new=64,lengths=(8,),warmup=0):
    assert_spec();out=pathlib.Path(out);out.mkdir(parents=True,exist_ok=True)
    target=load_model('8B');draft=load_model(scale)
    if adapter:draft=PeftModel.from_pretrained(draft,str(adapter),is_trainable=False)
    draft.eval();target.eval();eos=eos_ids(target)
    for r in records[:warmup]:
        ar(target,r['prompt_ids'],min(max_new,16),eos);speculative(target,draft,r['prompt_ids'],min(max_new,16),eos,lengths[0])
    results=[];failures=[]
    for r in records:
        baseline=ar(target,r['prompt_ids'],max_new,eos)
        for L in lengths:
            torch.cuda.reset_peak_memory_stats();candidate=speculative(target,draft,r['prompt_ids'],max_new,eos,L);mismatch=compare(baseline,candidate)
            entry={'id':r['id'],'domain':r['domain'],'scale':scale,'L':L,'max_new_tokens':max_new,'ar':baseline,'sd':candidate,'token_match':mismatch is None,'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved(),'at':now()}
            for name,value in [('ar',baseline),('sd',candidate)]:
                n=len(value['tokens']);value['committed_tokens_per_s']=n/(value['latency_ms']/1000)
                value['tpot_ms']=value['decode_seconds']*1000/(n-1) if n>1 else None
                value['decode_tokens_per_s']=(n-1)/value['decode_seconds'] if n>1 and value['decode_seconds']>0 else None
            rd=candidate['rounds'];drafted=sum(x['Lr'] for x in rd);accepted=sum(x['ar'] for x in rd)
            entry['draft_acceptance_rate']=accepted/drafted if drafted else None
            entry['mean_accepted_draft_length']=accepted/len(rd)
            entry['mean_committed_per_round']=sum(x['nr'] for x in rd)/len(rd)
            entry['local_speedup_vs_AR_diagnostic']=baseline['latency_ms']/candidate['latency_ms']
            append(out/'requests.jsonl',entry);results.append(entry)
            for n,rd in enumerate(candidate['rounds']):append(out/'rounds.jsonl',{'id':r['id'],'round':n,'scale':scale,'L':L,**rd})
            if mismatch:
                failure={'id':r['id'],'scale':scale,'L':L,**mismatch};append(out/'mismatches.jsonl',failure);failures.append(failure)
        print(json.dumps({'evaluated':len(results),'id':r['id'],'mismatches':len(failures),'scale':scale}),flush=True)
        # Preserve evidence and fail closed at the first request with any mismatch.
        if failures:break
    summary={'state':'passed' if not failures and len(results)==len(records)*len(lengths) else 'failed','expected_requests':len(records)*len(lengths),'completed_requests':len(results),'mismatches':len(failures),'prefix':sys.prefix,'scale':scale,'config':{'max_new':max_new,'lengths':list(lengths),'warmup':warmup},'provenance':provenance(),'speed_claims_allowed':False,'setting':'P0 diagnostic only'}
    atomic(out/'summary.json',summary);del target,draft;gc.collect();torch.cuda.empty_cache()
    assert summary['state']=='passed',f'SD correctness failure; see {out}/mismatches.jsonl'
    return summary

def select_checkpoint(scale,method,run):
    draft=load_model(scale);domains=DOMAINS if method.startswith('mix') else [method]
    rr=[r for d in domains for r in partition(d,'adapter_val')];tt=validate_teacher(rr)
    # Teacher max512 trajectories share the exact target greedy prefix. Only first256 is public reference.
    evidence=[]
    for epoch in (1,2):
        if epoch==1:draft=PeftModel.from_pretrained(draft,str(run/f'epoch{epoch}'),adapter_name='epoch1',is_trainable=False)
        else:draft.load_adapter(str(run/f'epoch{epoch}'),adapter_name='epoch2',is_trainable=False)
        draft.set_adapter(f'epoch{epoch}');draft.eval();by_domain={d:[] for d in domains}
        for r in tt:
            diag=public_prefix(draft,r['prompt_ids'],r['response_ids'][:256],8)
            append(ROOT/'results/validation'/f'{run.name}.{scale}.{method}.selection.jsonl',{'id':r['id'],'domain':r['domain'],'epoch':epoch,**diag})
            if diag['mean_a'] is not None:by_domain[r['domain']].append(diag['mean_a'])
        assert all(by_domain.values()),'No qualifying validation windows in a domain'
        means={d:statistics.mean(v) for d,v in by_domain.items()};evidence.append({'epoch':epoch,'domain_means':means,'macro_mean_a':statistics.mean(means.values()),'coverage':{d:len(v) for d,v in by_domain.items()}})
    best=max(evidence,key=lambda x:(x['macro_mean_a'],-x['epoch']))
    result={'selected_epoch':best['epoch'],'selection_rule':'macro request-level mean acceptance L8; ties earlier epoch','evidence':evidence,'adapter_path':str(run/f'epoch{best["epoch"]}'),'provenance':provenance(),'partition':'adapter_val'}
    atomic(run/'selected.json',result);del draft;gc.collect();torch.cuda.empty_cache();return result

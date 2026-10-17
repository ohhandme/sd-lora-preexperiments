"""Resume the frozen training protocol and audited checkpoint selection."""
import argparse, gc, pathlib, sys, statistics, math, os
ROOT = pathlib.Path('/root/autodl-tmp/sd-lora-preexp')
sys.path.insert(0, str(ROOT))
from spec_research.common import *
from spec_research.data import partition
from spec_research.teacher import validate_teacher
from spec_research.training import train
from spec_research.decoding import public_prefix
from spec_research.gate import validate_gate
from peft import PeftModel
import torch
RUN = 'overnight-20261017-seed42'
BASE = ROOT / 'results/continuation-20261017'

def validate_diagnostic(diag, reference, L=8):
    positions = [j for j in (0,32,64,128,192) if len(reference)-j >= L+1]
    assert [w['position'] for w in diag['windows']] == positions
    accepted = []
    for w in diag['windows']:
        assert len(w['proposal']) == L
        a = 0
        for x,y in zip(w['proposal'], reference[w['position']:]):
            if x != y: break
            a += 1
        assert w['a'] == a
        accepted.append(a)
    mean = statistics.mean(accepted) if accepted else None
    survival = [sum(a>=k for a in accepted)/len(accepted) for k in range(1,L+1)] if accepted else None
    assert diag['mean_a'] == mean and diag['survival'] == survival
    return {k:diag[k] for k in ('windows','mean_a','survival')}

def select_resume(scale, method, out):
    domains = DOMAINS if method.startswith('mix') else (method,)
    tt = validate_teacher([r for d in domains for r in partition(d,'adapter_val')])
    refs = {r['id']:r for r in tt}
    assert read(out/'run.json')['provenance']['source_hashes'] == sources()
    log = ROOT/'results/validation'/f'{RUN}.{scale}.{method}.selection.jsonl'
    cache_dir = out/'selection_cache'
    cache_dir.mkdir(exist_ok=True)
    manifest = dict(source_hashes=sources(), teacher_shas={r['id']:r['record_sha'] for r in tt},
        adapters={f'epoch{e}':digest(out/f'epoch{e}/adapter_model.safetensors') for e in (1,2)})
    mp = out/'selection_resume_manifest.json'
    if mp.exists(): assert read(mp) == manifest
    else: atomic(mp,manifest)
    cached = {}; logged = set()
    if log.exists():
        raw = log.read_bytes(); lines = raw.splitlines(keepends=True); good = []
        for i,line in enumerate(lines):
            try: row = __import__('json').loads(line)
            except __import__('json').JSONDecodeError:
                assert i == len(lines)-1 and not line.endswith(b'\n'), 'Interior corrupt selection record'
                archive = BASE/'recovery_archives'/f'{scale}.{method}.{digest(log)}.jsonl'
                archive.parent.mkdir(parents=True,exist_ok=True);archive.write_bytes(raw)
                log.write_bytes(b''.join(good));break
            good.append(line)
            key=(row['epoch'],row['id']); assert key[0] in (1,2) and key[1] in refs
            assert row['domain']==refs[key[1]]['domain']
            validate_diagnostic(row,refs[key[1]]['response_ids'][:256])
            if key in cached: assert cached[key] == row
            cached[key]=row;logged.add(key)
    for p in cache_dir.glob('*.json'):
        row=read(p);key=(row['epoch'],row['id']);assert key[0] in (1,2) and key[1] in refs
        validate_diagnostic(row,refs[key[1]]['response_ids'][:256])
        if key in cached: assert cached[key]==row
        cached[key]=row
    for key,row in cached.items():
        if key not in logged: append(log,row);logged.add(key)
    atomic(BASE/'selection_progress'/f'{scale}.{method}.json',dict(at=now(),reused=len(cached),expected=2*len(tt),state='auditing_cache'))
    draft=load_model(scale);evidence=[];audit_count=0
    for epoch in (1,2):
        if epoch==1: draft=PeftModel.from_pretrained(draft,str(out/'epoch1'),adapter_name='epoch1',is_trainable=False)
        else: draft.load_adapter(str(out/'epoch2'),adapter_name='epoch2',is_trainable=False)
        draft.set_adapter(f'epoch{epoch}');draft.eval()
        for domain in domains:
            audit=next((r for r in tt if r['domain']==domain and (epoch,r['id']) in cached),None)
            if audit:
                actual=public_prefix(draft,audit['prompt_ids'],audit['response_ids'][:256],8)
                expected=validate_diagnostic(cached[epoch,audit['id']],audit['response_ids'][:256])
                assert actual==expected, f'Cached selection replay mismatch: {epoch}/{audit["id"]}'
                audit_count+=1
        by_domain={d:[] for d in domains}
        for r in tt:
            key=(epoch,r['id'])
            if key not in cached:
                diag=public_prefix(draft,r['prompt_ids'],r['response_ids'][:256],8)
                validate_diagnostic(diag,r['response_ids'][:256])
                row=dict(id=r['id'],domain=r['domain'],epoch=epoch,**diag)
                atomic(cache_dir/f'{epoch}.{objhash(r["id"])}.json',row)
                append(log,row);cached[key]=row
            if cached[key]['mean_a'] is not None: by_domain[r['domain']].append(cached[key]['mean_a'])
            atomic(BASE/'selection_progress'/f'{scale}.{method}.json',dict(at=now(),completed=len(cached),expected=2*len(tt),audits_passed=audit_count,state='running'))
        assert all(by_domain.values())
        means={d:statistics.mean(v) for d,v in by_domain.items()}
        evidence.append(dict(epoch=epoch,domain_means=means,macro_mean_a=statistics.mean(means.values()),coverage={d:len(v) for d,v in by_domain.items()}))
    best=max(evidence,key=lambda x:(x['macro_mean_a'],-x['epoch']))
    result=dict(selected_epoch=best['epoch'],selection_rule='macro request-level mean acceptance L8; ties earlier epoch',evidence=evidence,adapter_path=str(out/f'epoch{best["epoch"]}'),provenance=provenance(),partition='adapter_val')
    assert len(cached)==2*len(tt);atomic(out/'selected.json',result)
    atomic(BASE/'selection_progress'/f'{scale}.{method}.json',dict(at=now(),completed=len(cached),expected=2*len(tt),audits_passed=audit_count,state='complete'))
    del draft;gc.collect();torch.cuda.empty_cache();return result

def training(scale,method):
    validate_gate();domains=DOMAINS if method.startswith('mix') else (method,)
    tt=validate_teacher([r for d in domains for r in partition(d,'expert_train')])
    out=ROOT/'adapters'/scale/method/'seed42'/RUN
    train(scale,method,tt,out,ROOT/'logs/train'/f'{RUN}.{scale}.{method}.jsonl')
    if (out/'selected.json').exists():
        selection=read(out/'selected.json');assert selection['provenance']['source_hashes']==sources()
    else: selection=select_resume(scale,method,out)
    c=read(out/'complete.json');assert len(c['exposure_counts'])==len(tt) and set(c['exposure_counts'].values())=={2}
    atomic(BASE/'jobs'/f'{scale}.{method}.json',dict(state='complete',scale=scale,method=method,adapter=str(out),selected=selection,at=now()))

if __name__=='__main__':
    assert_spec();torch.set_num_threads(4)
    p=argparse.ArgumentParser();p.add_argument('--scale',required=True);p.add_argument('--method',required=True);p.add_argument('--managed-lock-fd',type=int,required=True);a=p.parse_args()
    with gpu_lock(a.managed_lock_fd):training(a.scale,a.method)

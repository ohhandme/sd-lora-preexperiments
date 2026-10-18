"""Frozen data access and diagnostics for the sequential v5 pre-experiment."""
import collections
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import statistics as st
import subprocess
import sys
import time

ROOT=Path('/root/autodl-tmp/sd-lora-preexp')
EXP=Path(__file__).resolve().parent
V2=ROOT/'experiments/v2-routing'
V4=ROOT/'experiments/v4-complementarity'
OUT=EXP/'results'
LEDGER=ROOT/'autoresearch/loop-v5-sequential-20261018'
sys.path[:0]=[str(ROOT),str(V2/'scripts')]
import numpy as np
import torch
from peft import PeftModel
from scipy.optimize import linear_sum_assignment
from spec_research.common import assert_spec,atomic,digest,load_model,now,objhash,seed_all
from spec_research.decoding import CachedModel,top
from train import DOMAINS,records

POS=(0,64)
POOLS=('semantic','random','mismatch')

def read(path): return json.loads(Path(path).read_text())
def mean(x):
    x=list(x)
    return float(st.mean(x)) if x else None
def key(row): return hashlib.sha256(row['id'].encode()).hexdigest()+'.json'
def selected(method): return Path(read(V2/'adapters/0.6B'/method/'seed42/selected.json')['adapter_path'])
def status(stage,**kw):
    item={'at':now(),'pid':os.getpid(),'stage':stage,**kw}
    atomic(EXP/'status/current.json',item)
    print(json.dumps(item),flush=True)
def gpu_idle():
    out=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
    foreign=[int(x) for x in out.splitlines() if x.strip().isdigit() and int(x)!=os.getpid()]
    assert not foreign,f'Other GPU processes: {foreign}; not disturbing them'
    import shutil
    assert shutil.disk_usage(ROOT).free>2*1024**3,'Less than 2 GiB free'
def cleanup(*args):
    gc.collect();torch.cuda.empty_cache()
def boot():
    assert_spec();torch.set_num_threads(4);seed_all(42)
def protected():
    files=[ROOT/'program.md']
    for folder in (ROOT/'spec_research',ROOT/'configs',V2,ROOT/'experiments/v3-async-budget',V4):
        files.extend(p for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    return {str(p):digest(p) for p in sorted(set(files))}
def protect_check():
    old=read(OUT/'protected.json')
    changed=[p for p,h in old.items() if not Path(p).is_file() or digest(p)!=h]
    assert not changed,changed[:10]
    return len(old)

def sorted_rows(rows,seed):
    return sorted(rows,key=lambda r:hashlib.sha256((seed+'|'+r['id']).encode()).hexdigest())
def data():
    expert=records('mix16','expert_train'); source_route=records('mix16','router_train')
    val=records('mix16','adapter_val'); evaluation=records('mix16','router_val')
    selected_train=[]; fit=[]; cv=[]
    for domain in DOMAINS:
        group=sorted_rows([r for r in expert if r['domain']==domain],'v4-complementarity-smoke-20261018')
        selected_train.extend(group[:32])
        fit.extend(sorted_rows(group[32:],'v5-route-20261018')[:40])
        cv.extend(sorted_rows([r for r in val if r['domain']==domain],'v5-val-20261018')[:10])
    splits={'train':source_route+selected_train,'route':fit,'val':cv,'eval':evaluation}
    all_rows=[r for rr in splits.values() for r in rr]
    assert len(all_rows)==1010==len({r['id'] for r in all_rows})
    assert len({objhash(r['prompt_ids']) for r in all_rows})==len(all_rows),'Duplicate prompt across new partitions'
    manifest={'schema':1,'program_sha':digest(EXP/'program.md'),'common_sha':digest(__file__),
              'L':8,'positions':list(POS),'max_response':128,
              'source_weights':{m:digest(selected(m)/'adapter_model.safetensors') for m in (*DOMAINS,'mix16','mix80')},
              'splits':{s:[{'id':r['id'],'domain':r['domain'],'original_partition':r['partition'],
                            'teacher_sha':r['record_sha'],'prompt_sha':objhash(r['prompt_ids'])} for r in rr] for s,rr in splits.items()}}
    path=OUT/'input.json'
    if path.exists(): assert read(path)==manifest,'Frozen input changed'
    else: atomic(path,manifest)
    return splits,objhash(manifest)

def validate(diag,row):
    positions=[p for p in POS if len(row['response_ids'][:128])-p>=9]
    assert [w['position'] for w in diag['windows']]==positions
    for w in diag['windows']:
        reference=row['response_ids'][w['position']:w['position']+8]
        assert 1<=len(w['proposal'])<=8
        a=next((j for j,(x,y) in enumerate(zip(w['proposal'],reference)) if x!=y),len(w['proposal']))
        assert a==w['a'] and (a< len(w['proposal']) or a==8)
    assert diag['mean_a']==mean(w['a'] for w in diag['windows'])
    return diag

@torch.inference_mode()
def diagnostic(model,row):
    windows=[]
    for p in POS:
        if len(row['response_ids'][:128])-p<9: continue
        c=CachedModel(model); context=row['prompt_ids']+row['response_ids'][:p]
        proposal=[];a=0
        for j in range(8):
            token=top(c.next(context));proposal.append(token);context.append(token)
            if token!=row['response_ids'][p+j]: break
            a+=1
        windows.append({'position':p,'a':a,'proposal':proposal})
    return validate({'windows':windows,'mean_a':mean(w['a'] for w in windows)},row)

def cached_legacy(row,method):
    name=key(row)
    candidates=[V2/'results/routing/diagnostics'/method/name,V2/'results/quality/matrix'/method/name]
    candidates += [V4/'results/matrix'/s/('old_'+method)/name for s in ('train','route','eval')]
    if method=='mix80': candidates.append(ROOT/'experiments/v3-async-budget/results/mix80_router_train'/name)
    for path in candidates:
        if not path.exists():continue
        item=read(path);assert item['id']==row['id']
        windows=[w for w in item['diagnostic']['windows'] if w['position'] in POS and len(row['response_ids'][:128])-w['position']>=9]
        windows=[{**w,'proposal':w['proposal'][:min(w['a']+1,8)]} for w in windows]
        return validate({'windows':windows,'mean_a':mean(w['a'] for w in windows)},row),str(path)
    return None,None

def matrix(rows,name,stage,adapter,binding,legacy=None):
    result={};model=None;reused=0;generated=0
    gpu_idle()
    for index,row in enumerate(rows,1):
        path=OUT/'matrix'/stage/name/key(row)
        if path.exists():
            item=read(path);assert item['binding']==binding and item['id']==row['id']
            diag=validate(item['diagnostic'],row)
        else:
            diag,origin=cached_legacy(row,legacy) if legacy else (None,None)
            if diag is not None:reused+=1
            else:
                if model is None:
                    model=PeftModel.from_pretrained(load_model('0.6B'),adapter,is_trainable=False).eval()
                diag=diagnostic(model,row);generated+=1;origin='v5 greedy first-rejection computation'
            atomic(path,{'id':row['id'],'domain':row['domain'],'binding':binding,'diagnostic':diag,
                         'origin':origin,'adapter_sha':digest(adapter/'adapter_model.safetensors')})
        result[row['id']]=diag
        if index%25==0 or index==len(rows):status('matrix',method=name,partition=stage,done=index,total=len(rows),reused=reused,generated=generated)
    del model;cleanup()
    return result

def load_matrix(rows,name,stage,binding):
    result={}
    for row in rows:
        item=read(OUT/'matrix'/stage/name/key(row));assert item['binding']==binding and item['id']==row['id']
        result[row['id']]=validate(item['diagnostic'],row)
    return result

def source(splits,binding):
    for m in (*DOMAINS,'mix16'):
        matrix(splits['train'],'old_'+m,'train',selected(m),binding,legacy=m)
    for stage in ('val','eval'):
        for m in ('mix16','mix80'):
            matrix(splits[stage],'old_'+m,stage,selected(m),binding,legacy=m)
    rows=splits['train']
    old_tables={m:load_matrix(rows,'old_'+m,'train',binding) for m in DOMAINS}
    a=np.array([[old_tables[m][r['id']]['mean_a'] or 0. for m in DOMAINS] for r in rows])
    columns=np.repeat(np.arange(5),128)
    r,c=linear_sum_assignment(-(a-a.mean(axis=1,keepdims=True))[:,columns])
    mismatch={rows[i]['id']:int(columns[j]) for i,j in zip(r,c)}
    order=list(range(640));random.Random(42).shuffle(order)
    maps={'semantic':{x['id']:DOMAINS.index(x['domain']) for x in rows},
          'random':{rows[i]['id']:j//128 for j,i in enumerate(order)},'mismatch':mismatch}
    assert all(sorted(collections.Counter(m.values()).values())==[128]*5 for m in maps.values())
    result={'maps':maps,'row_ids':[r['id'] for r in rows],'scores':a.tolist(),
            'all_tie':int(sum(len(set(v))==1 for v in a)),
            'unique_best':int(sum(sum(v==max(v))==1 for v in a)),
            'semantic_agreement':mean(mismatch[k]==maps['semantic'][k] for k in mismatch),
            'reassigned_strict_gain':int(sum(a[i,mismatch[r['id']]]>a[i,maps['semantic'][r['id']]] for i,r in enumerate(rows)))}
    atomic(OUT/'assignments.json',result)
    status('source_complete',semantic_agreement=result['semantic_agreement'],unique_best=result['unique_best'])

def paired(a,b):
    d=np.array(a)-np.array(b);assert len(d) and np.isfinite(d).all()
    rng=np.random.default_rng(42)
    boot=rng.choice(d,(2000,len(d)),replace=True).mean(axis=1)
    return {'mean':float(d.mean()),'ci95':np.quantile(boot,[.025,.975]).tolist(),'requests':len(d)}

def values(rows,tables):
    keys=[(r['id'],w['position']) for r in rows for w in tables[0][r['id']]['windows']]
    lookups=[{(r['id'],w['position']):w['a'] for r in rows for w in t[r['id']]['windows']} for t in tables]
    assert all(set(t)==set(keys) for t in lookups)
    return keys,np.array([[t[k] for t in lookups] for k in keys],dtype=float)
def aggregate(keys,vector):
    ids=sorted({k[0] for k in keys})
    return {rid:mean(v for k,v in zip(keys,vector) if k[0]==rid) for rid in ids}
def pool_stats(rows,tables,fixed=None):
    keys,a=values(rows,tables);ids=sorted({k[0] for k in keys})
    per={str(i):aggregate(keys,a[:,i]) for i in range(a.shape[1])}
    means=[mean(v.values()) for v in per.values()]
    if fixed is None:fixed=int(np.argmax(means))
    request_oracle={rid:max(per[str(i)][rid] for i in range(a.shape[1])) for rid in ids}
    prefix_oracle=aggregate(keys,a.max(axis=1))
    return {'requests':len(ids),'windows':len(keys),'expert_means':means,'fixed_index':fixed,
            'fixed':mean(per[str(fixed)].values()),'request_oracle':mean(request_oracle.values()),
            'prefix_oracle':mean(prefix_oracle.values()),'different_acceptance_windows':int(sum(np.ptp(a,axis=1)>0)),
            'unique_best_windows':int(sum((a==a.max(axis=1)[:,None]).sum(axis=1)==1)),
            'all_early_reject_windows':int(sum((a<4).all(axis=1))),
            'removal_losses':[float(np.mean(a.max(axis=1)-np.delete(a,i,axis=1).max(axis=1))) for i in range(a.shape[1])] if a.shape[1]>1 else [],
            'per_request':{rid:{'fixed':per[str(fixed)][rid],'request_oracle':request_oracle[rid],
                                'prefix_oracle':prefix_oracle[rid]} for rid in ids}}

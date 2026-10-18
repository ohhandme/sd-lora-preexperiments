#!/root/miniconda3/envs/spec/bin/python
"""Small matched v2 online baseline; one isolated process per configuration."""

import argparse
import hashlib
import json
import os
import pathlib
import pickle
import sys
import time

ROOT = pathlib.Path('/root/autodl-tmp/sd-lora-preexp')
V2 = ROOT/'experiments/v2-routing'
sys.path[:0] = [str(ROOT),str(V2/'scripts')]

import torch
from peft import PeftModel
from spec_research.common import assert_spec, atomic, digest, load_model, now, objhash
from spec_research.decoding import ar, speculative, compare
from train import DOMAINS, records

OUT = V2/'results/online_pilot'
METHODS = ('AR8','BASE','mix16','mix80','DOMAIN','ROUTE-TASK','ROUTE-ACC')


def subset():
    all_rows = records('mix16','router_val')
    selection = []
    for domain in DOMAINS:
        domain_rows = [r for r in all_rows if r['domain'] == domain]
        domain_rows.sort(key=lambda r: hashlib.sha256(('v2-online-pilot-20261017|'+r['id']).encode()).hexdigest())
        selection.extend(domain_rows[:8])
    assert len(selection) == 40 and len({r['id'] for r in selection}) == 40
    binding = {'seed': 'v2-online-pilot-20261017', 'ids': [r['id'] for r in selection],
               'teacher_shas': [r['record_sha'] for r in selection],
               'prompt_sha256': digest(V2/'data/processed/pilot_manifest.jsonl'),
               'policy': 'BF16 B1 max_new256 L8; 8 requests per task; single timing repeat'}
    path = OUT/'subset_manifest.json'
    if path.exists():
        assert json.loads(path.read_text()) == binding
    else:
        atomic(path,binding)
    return selection, objhash(binding)


def selected(method):
    p = V2/'adapters/0.6B'/method/'seed42/selected.json'
    x = json.loads(p.read_text())
    assert x['partition'] == 'adapter_val'
    return x['adapter_path']


def baseline_path(row):
    return OUT/'AR8'/(hashlib.sha256(row['id'].encode()).hexdigest()+'.json')


def output_path(method,row):
    return OUT/method/(hashlib.sha256(row['id'].encode()).hexdigest()+'.json')


def eos(model):
    value = model.generation_config.eos_token_id
    return set([value] if isinstance(value,int) else value)


def load_router(name):
    source = V2/'results/routing'/('task_router.pkl' if name == 'ROUTE-TASK'
                                    else 'acceptance_router.pkl')
    with source.open('rb') as f:
        return pickle.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method',choices=METHODS,required=True)
    args = parser.parse_args()
    assert_spec()
    torch.set_num_threads(4)
    rows, subset_sha = subset()
    prompt_map = {x['record_id']:x['prompt'] for x in
                  (json.loads(line) for line in (V2/'data/processed/pilot_manifest.jsonl').read_text().splitlines()
                   if line)}
    target = load_model('8B')
    target.eval()
    draft = None
    router = None
    method = args.method
    if method != 'AR8':
        draft = load_model('0.6B')
        if method in ('mix16','mix80'):
            draft = PeftModel.from_pretrained(draft,selected(method),adapter_name=method,
                                               is_trainable=False)
        elif method in ('DOMAIN','ROUTE-TASK','ROUTE-ACC'):
            for index,domain in enumerate(DOMAINS):
                if index == 0:
                    draft = PeftModel.from_pretrained(draft,selected(domain),adapter_name=domain,
                                                       is_trainable=False)
                else:
                    draft.load_adapter(selected(domain),adapter_name=domain,is_trainable=False)
        if method in ('ROUTE-TASK','ROUTE-ACC'):
            router = load_router(method)
        draft.eval()
    eos_ids = eos(target)
    # Warm-up requests are disjoint from the 40 timed router_val requests.
    warm_rows = records('code','expert_train')[:5]
    for row in warm_rows:
        if method == 'AR8':
            ar(target,row['prompt_ids'],16,eos_ids)
        else:
            if method in ('DOMAIN','ROUTE-TASK','ROUTE-ACC'):
                draft.set_adapter('code')
            speculative(target,draft,row['prompt_ids'],16,eos_ids,8)
    progress = V2/'status'/f'online.{method}.json'
    completed = 0
    for row in rows:
        path = output_path(method,row)
        if path.exists():
            saved = json.loads(path.read_text())
            assert saved['id'] == row['id'] and saved['subset_sha'] == subset_sha
            completed += 1
            continue
        torch.cuda.reset_peak_memory_stats()
        if method == 'AR8':
            result = ar(target,row['prompt_ids'],256,eos_ids)
            choice = None
            routing_ms = 0.0
        else:
            baseline = json.loads(baseline_path(row).read_text())
            assert baseline['subset_sha'] == subset_sha and baseline['id'] == row['id']
            choice = row['domain'] if method == 'DOMAIN' else None
            routing_ms = 0.0
            if router is not None:
                start = time.perf_counter()
                choice = str(router.predict([prompt_map[row['id']]])[0])
                routing_ms = (time.perf_counter()-start)*1000
            if choice is not None:
                assert choice in DOMAINS
                draft.set_adapter(choice)
            result = speculative(target,draft,row['prompt_ids'],256,eos_ids,8)
            mismatch = compare(baseline['result'],result)
            if mismatch is not None:
                failure = {'state':'failed','at':now(),'id':row['id'],'method':method,
                           'baseline':baseline['result'],'candidate':result,'mismatch':mismatch}
                atomic(OUT/'failures'/method/(hashlib.sha256(row['id'].encode()).hexdigest()+'.json'),failure)
                raise RuntimeError('Full token mismatch; no request committed')
        token_ids = result['tokens']
        item = {'id':row['id'],'domain':row['domain'],'method':method,
                'subset_sha':subset_sha,'choice':choice,'routing_ms':routing_ms,
                'result':result,'eos':token_ids[-1] in eos_ids,
                'truncated':len(token_ids)==256 and token_ids[-1] not in eos_ids,
                'peak_allocated':torch.cuda.max_memory_allocated(),
                'peak_reserved':torch.cuda.max_memory_reserved(),
                'token_eos_truncation_equal':True if method == 'AR8' else
                    (token_ids == baseline['result']['tokens'] and
                     (token_ids[-1] in eos_ids) == baseline['eos'] and
                     (len(token_ids)==256 and token_ids[-1] not in eos_ids) == baseline['truncated'])}
        assert item['token_eos_truncation_equal']
        atomic(path,item)
        completed += 1
        atomic(progress,{'state':'running','at':now(),'pid':os.getpid(),
                         'method':method,'completed':completed,'requested':40})
        print(json.dumps({'method':method,'completed':completed,'requested':40}),flush=True)
    atomic(progress,{'state':'complete','at':now(),'method':method,
                     'completed':completed,'requested':40})


if __name__ == '__main__':
    main()

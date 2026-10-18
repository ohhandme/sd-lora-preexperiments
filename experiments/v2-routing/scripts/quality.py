#!/root/miniconda3/envs/spec/bin/python
"""V2 pilot L8 validation matrix and offline expert headroom only."""

import collections
import contextlib
import hashlib
import json
import os
import pathlib
import random
import statistics
import sys

ROOT = pathlib.Path('/root/autodl-tmp/sd-lora-preexp')
V2 = ROOT / 'experiments/v2-routing'
sys.path[:0] = [str(ROOT), str(V2/'scripts')]

import torch
from peft import PeftModel
from spec_research.common import assert_spec, atomic, digest, load_model, now, objhash
from spec_research.decoding import public_prefix
from train import DOMAINS, METHODS, records

OUT = V2 / 'results/quality'
SELECTED = {name: V2/'adapters/0.6B'/name/'seed42/selected.json' for name in METHODS}


def mean(values):
    return statistics.mean(values) if values else None


def validate_diag(diag, reference):
    positions = [j for j in (0,32,64,128,192) if len(reference)-j >= 9]
    assert [w['position'] for w in diag['windows']] == positions
    accepted = []
    for window in diag['windows']:
        assert len(window['proposal']) == 8
        a = 0
        for x,y in zip(window['proposal'], reference[window['position']:]):
            if x != y:
                break
            a += 1
        assert window['a'] == a
        accepted.append(a)
    expected = mean(accepted)
    assert diag['mean_a'] == expected
    assert diag['survival'] == ([sum(a >= k for a in accepted)/len(accepted)
                                 for k in range(1,9)] if accepted else None)
    return diag


def input_binding(rows):
    selected = {m: json.loads(SELECTED[m].read_text()) for m in METHODS}
    binding = {'scale': '0.6B', 'L': 8, 'partition': 'adapter_val',
               'pilot_sha256': digest(V2/'data/processed/pilot_manifest.jsonl'),
               'ids': [r['id'] for r in rows],
               'teacher_shas': {r['id']: r['record_sha'] for r in rows},
               'adapter_shas': {m: digest(pathlib.Path(selected[m]['adapter_path'])/'adapter_model.safetensors')
                                for m in METHODS}, 'selection_epochs':
               {m: selected[m]['selected_epoch'] for m in METHODS},
               'policy': 'BF16 B1 L8 public target prefix positions 0/32/64/128/192, first256'}
    path = OUT / 'input_manifest.json'
    if path.exists():
        assert json.loads(path.read_text()) == binding
    else:
        atomic(path, binding)
    return selected, objhash(binding)


def pool(selected):
    model = load_model('0.6B')
    for index, method in enumerate(METHODS):
        path = selected[method]['adapter_path']
        if index == 0:
            model = PeftModel.from_pretrained(model, path, adapter_name=method,
                                               is_trainable=False)
        else:
            model.load_adapter(path, adapter_name=method, is_trainable=False)
    return model.eval()


def bootstrap_delta(rows, first, second, seed=42, repeats=2000):
    differences = [r[first]-r[second] for r in rows]
    rng = random.Random(seed)
    estimates = []
    for _ in range(repeats):
        estimates.append(sum(differences[rng.randrange(len(differences))]
                             for _ in differences)/len(differences))
    estimates.sort()
    return {'mean': mean(differences), 'ci95': [estimates[int(.025*repeats)],
                                             estimates[int(.975*repeats)-1]],
            'n': len(differences), 'bootstrap_repeats': repeats}


def report(rows, binding):
    indexed = {}
    for method in ('BASE', *METHODS):
        for r in rows:
            path = OUT/'matrix'/method/(hashlib.sha256(r['id'].encode()).hexdigest()+'.json')
            if path.exists():
                item = json.loads(path.read_text())
                assert item['id'] == r['id'] and item['binding'] == binding
                validate_diag(item['diagnostic'], r['response_ids'][:256])
                indexed[(r['id'],method)] = item['diagnostic']
    complete = [r for r in rows if all((r['id'],m) in indexed for m in ('BASE', *METHODS))]
    qualified = [r for r in complete if indexed[(r['id'],'BASE')]['mean_a'] is not None]
    assert all(indexed[(r['id'],m)]['mean_a'] is not None
               for r in qualified for m in ('BASE', *METHODS))
    values = []
    for r in qualified:
        scores = {m: indexed[(r['id'],m)]['mean_a'] for m in ('BASE', *METHODS)}
        scores['DOMAIN'] = scores[r['domain']]
        scores['ORACLE'] = max(scores[m] for m in DOMAINS)
        values.append({'id': r['id'], 'domain': r['domain'], **scores})
    fixed_scores = {m: mean([x[m] for x in values]) for m in DOMAINS}
    fixed = max(DOMAINS, key=lambda m: fixed_scores[m]) if values else None
    if fixed:
        for item in values:
            item['FIXEDBEST'] = item[fixed]
    names = ('BASE', *METHODS, 'DOMAIN', 'FIXEDBEST', 'ORACLE')
    means = {name: {'macro': mean([x[name] for x in values]),
                    'by_domain': {d: mean([x[name] for x in values if x['domain'] == d])
                                  for d in DOMAINS}} for name in names} if values else {}
    paired = {f'ORACLE_minus_{reference}': bootstrap_delta(values, 'ORACLE', reference)
              for reference in ('mix16','mix80','FIXEDBEST')} if values else {}
    paired.update({f'DOMAIN_minus_{reference}': bootstrap_delta(values, 'DOMAIN', reference)
                   for reference in ('mix16','mix80')} if values else {})
    result = {'state': 'complete' if len(complete) == 240 else 'partial',
              'at': now(), 'partition': 'adapter_val', 'scale': '0.6B', 'L': 8,
              'binding': binding, 'expected_requests': 240,
              'complete_requests': len(complete), 'qualified_requests': len(qualified),
              'excluded_short_response': len(complete)-len(qualified),
              'expected_entries': 240*8, 'completed_entries': len(indexed),
              'fixed_best': fixed, 'means': means, 'paired': paired,
              'quality_headroom_only': all(v['ci95'][0] > 0 for k,v in paired.items()
                                           if k.startswith('ORACLE_minus_')) if paired else None,
              'caveat': 'Oracle is offline retrospective acceptance only; no online speed or cost gate measured.'}
    atomic(OUT/'summary.json', result)
    if result['state'] == 'complete':
        atomic(OUT/'per_request.json', {'binding': binding, 'requests': values})
    return result


def main():
    assert_spec()
    torch.set_num_threads(4)
    rows = records('mix16', 'adapter_val')
    assert len(rows) == 240
    selected, binding = input_binding(rows)
    model = pool(selected)
    completed = 0
    for method in ('BASE', *METHODS):
        if method != 'BASE':
            model.set_adapter(method)
        for r in rows:
            path = OUT/'matrix'/method/(hashlib.sha256(r['id'].encode()).hexdigest()+'.json')
            if path.exists():
                item = json.loads(path.read_text())
                assert item['id'] == r['id'] and item['binding'] == binding
                validate_diag(item['diagnostic'], r['response_ids'][:256])
                completed += 1
                continue
            cm = model.disable_adapter() if method == 'BASE' else contextlib.nullcontext()
            with cm:
                diag = public_prefix(model, r['prompt_ids'], r['response_ids'][:256], 8)
            validate_diag(diag, r['response_ids'][:256])
            atomic(path, {'id': r['id'], 'domain': r['domain'], 'method': method,
                          'L': 8, 'binding': binding, 'diagnostic': diag,
                          'origin': 'GPU public-prefix greedy proposal'})
            completed += 1
            if completed % 10 == 0:
                atomic(V2/'status/quality.json', {'state': 'running', 'at': now(),
                                                  'pid': os.getpid(), 'completed': completed,
                                                  'expected': 1920, 'method': method})
                print(json.dumps({'completed': completed, 'expected': 1920,
                                  'method': method}), flush=True)
    result = report(rows, binding)
    atomic(V2/'status/quality.json', {'state': result['state'], 'at': now(),
                                      'completed': result['completed_entries'],
                                      'expected': 1920})
    assert result['state'] == 'complete'
    print(json.dumps({'state': 'complete', 'headroom': result['paired']}), flush=True)


if __name__ == '__main__':
    main()

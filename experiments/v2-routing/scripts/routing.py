#!/root/miniconda3/envs/spec/bin/python
"""V2 pilot acceptance labels and prompt-only routing baselines."""

import hashlib
import json
import os
import pathlib
import pickle
import statistics
import sys
import time

ROOT = pathlib.Path('/root/autodl-tmp/sd-lora-preexp')
V2 = ROOT / 'experiments/v2-routing'
sys.path[:0] = [str(ROOT), str(V2/'scripts')]

import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from spec_research.common import assert_spec, atomic, digest, now, objhash
from spec_research.decoding import public_prefix
from train import DOMAINS, records
from quality import METHODS, pool, validate_diag

OUT = V2 / 'results/routing'


def prompt_lookup():
    rows = [json.loads(line) for line in (V2/'data/processed/pilot_manifest.jsonl').read_text().splitlines() if line]
    return {r['record_id']: r['prompt'] for r in rows}


def label(scores, domain):
    best = max(scores.values())
    tied = [d for d in DOMAINS if scores[d] == best]
    return domain if domain in tied else tied[0]


def atomic_pickle(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+f'.{os.getpid()}.tmp')
    with temp.open('wb') as f:
        pickle.dump(value, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def make_clf():
    return make_pipeline(TfidfVectorizer(ngram_range=(1,2), max_features=20000,
                                         min_df=2, sublinear_tf=True),
                         LogisticRegression(C=1.0, max_iter=1000,
                                            class_weight='balanced', random_state=42))


def methods_for(row):
    return (*DOMAINS, 'mix16', 'mix80') if row['partition'] == 'router_val' else DOMAINS


def main():
    assert_spec()
    torch.set_num_threads(4)
    train_rows = records('mix16', 'router_train')
    val_rows = records('mix16', 'router_val')
    rows = train_rows + val_rows
    assert len(rows) == 600 and len({r['id'] for r in rows}) == 600
    prompts = prompt_lookup()
    selected = {m: json.loads((V2/'adapters/0.6B'/m/'seed42/selected.json').read_text())
                for m in METHODS}
    binding = {'pilot_sha256': digest(V2/'data/processed/pilot_manifest.jsonl'),
               'router_ids': [r['id'] for r in rows],
               'teacher_shas': {r['id']: r['record_sha'] for r in rows},
               'adapter_shas': {m: digest(pathlib.Path(selected[m]['adapter_path'])/'adapter_model.safetensors')
                                for m in METHODS},
               'policy': 'BF16 L8 first256 public-prefix, five experts for train labels; mixed references on val'}
    binding_path = OUT/'input_manifest.json'
    if binding_path.exists():
        assert json.loads(binding_path.read_text()) == binding
    else:
        atomic(binding_path, binding)
    fingerprint = objhash(binding)
    model = pool(selected)
    total = 0
    for r in rows:
        for method in methods_for(r):
            path = OUT/'diagnostics'/method/(hashlib.sha256(r['id'].encode()).hexdigest()+'.json')
            if path.exists():
                item = json.loads(path.read_text())
                assert item['id'] == r['id'] and item['fingerprint'] == fingerprint
                validate_diag(item['diagnostic'], r['response_ids'][:256])
            else:
                model.set_adapter(method)
                diag = validate_diag(public_prefix(model, r['prompt_ids'],
                                                   r['response_ids'][:256], 8), r['response_ids'][:256])
                atomic(path, {'id': r['id'], 'partition': r['partition'],
                              'method': method, 'diagnostic': diag,
                              'fingerprint': fingerprint})
            total += 1
        if total % 25 == 0:
            atomic(V2/'status/routing.json', {'state': 'running', 'at': now(),
                                               'pid': os.getpid(), 'diagnostics': total,
                                               'expected_diagnostics': 3240})
            print(json.dumps({'diagnostics': total, 'expected': 3240}), flush=True)
    diagnostics = {}
    for r in rows:
        item = {}
        for method in methods_for(r):
            path = OUT/'diagnostics'/method/(hashlib.sha256(r['id'].encode()).hexdigest()+'.json')
            diag = json.loads(path.read_text())['diagnostic']
            item[method] = diag['mean_a']
        diagnostics[r['id']] = item
    qualified_train = [r for r in train_rows if all(diagnostics[r['id']][d] is not None for d in DOMAINS)]
    qualified_val = [r for r in val_rows if all(diagnostics[r['id']][m] is not None
                      for m in (*DOMAINS,'mix16','mix80'))]
    if len(qualified_train) < 400 or len(qualified_val) < 100:
        raise RuntimeError('Routing label coverage too low; preserve short cases and diagnose')
    task = make_clf()
    task.fit([prompts[r['id']] for r in train_rows], [r['domain'] for r in train_rows])
    acc = make_clf()
    labels = [label({d: diagnostics[r['id']][d] for d in DOMAINS}, r['domain'])
              for r in qualified_train]
    if len(set(labels)) < 2:
        raise RuntimeError('Only one acceptance-best expert class: do not train degenerate router')
    acc.fit([prompts[r['id']] for r in qualified_train], labels)
    atomic_pickle(OUT/'task_router.pkl', task)
    atomic_pickle(OUT/'acceptance_router.pkl', acc)
    comparison = []
    for r in qualified_val:
        values = diagnostics[r['id']]
        experts = {d: values[d] for d in DOMAINS}
        best = max(experts.values())
        t0 = time.perf_counter()
        task_choice = str(task.predict([prompts[r['id']]])[0])
        task_ms = (time.perf_counter()-t0)*1000
        t0 = time.perf_counter()
        acc_choice = str(acc.predict([prompts[r['id']]])[0])
        acc_ms = (time.perf_counter()-t0)*1000
        comparison.append({'id': r['id'], 'domain': r['domain'],
                           'best_label': label(experts, r['domain']),
                           'domain_known': experts[r['domain']],
                           'task_choice': task_choice, 'task_score': experts[task_choice],
                           'task_ms': task_ms, 'acc_choice': acc_choice,
                           'acc_score': experts[acc_choice], 'acc_ms': acc_ms,
                           'mix16': values['mix16'], 'mix80': values['mix80'],
                           'oracle': best,
                           'task_optimal_hit': experts[task_choice] == best,
                           'acc_optimal_hit': experts[acc_choice] == best})
    def average(key):
        return statistics.mean(item[key] for item in comparison)
    result = {'state': 'complete', 'at': now(), 'fingerprint': fingerprint,
              'router_train_count': len(train_rows), 'router_val_count': len(val_rows),
              'qualified_train': len(qualified_train), 'qualified_val': len(qualified_val),
              'excluded_train_short': len(train_rows)-len(qualified_train),
              'excluded_val_short': len(val_rows)-len(qualified_val),
              'task_domain_accuracy': statistics.mean(x['task_choice'] == x['domain'] for x in comparison),
              'task_optimal_set_hit': average('task_optimal_hit'),
              'acceptance_optimal_set_hit': average('acc_optimal_hit'),
              'mean_a': {name: average(name) for name in
                         ('domain_known','task_score','acc_score','mix16','mix80','oracle')},
              'router_ms': {'task': average('task_ms'), 'acceptance': average('acc_ms')},
              'caveat': 'Offline prompt-only routing quality; online target calls and switch costs not included.'}
    atomic(OUT/'summary.json', result)
    atomic(OUT/'per_request.json', {'fingerprint': fingerprint, 'requests': comparison})
    atomic(V2/'status/routing.json', {'state': 'complete', 'at': now(),
                                      'diagnostics': total, 'expected_diagnostics': 3240})
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()

#!/root/miniconda3/envs/spec/bin/python
"""P2 top-2 correction branch feasibility on frozen request-heldout prefixes.

No target model is run: teacher tokens from V2 are used only *after* the draft
action to score the branch. Results are retrospective diagnostics, not speed.
"""

import hashlib
import json
import os
import pathlib
import sys
import time

ROOT = pathlib.Path('/root/autodl-tmp/sd-lora-preexp')
V2 = ROOT / 'experiments/v2-routing'
V3 = ROOT / 'experiments/v3-async-budget'
sys.path[:0] = [str(ROOT), str(V2 / 'scripts'), str(V3 / 'scripts')]

import numpy as np
import torch
from peft import PeftModel
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from spec_research.common import assert_spec, atomic, digest, load_model, now
from spec_research.decoding import CachedModel, sync
from train import DOMAINS, records
from offline_probe import SEED, diagnostic, feature, split_ids

OUT = V3 / 'results/branch_probe_v2'


def selected(method):
    info = json.loads((V2 / 'adapters/0.6B' / method / 'seed42/selected.json').read_text())
    assert info['partition'] == 'adapter_val'
    return info['adapter_path']


def model_and_policy(rows, split):
    model = load_model('0.6B')
    for k, domain in enumerate(DOMAINS):
        if k == 0:
            model = PeftModel.from_pretrained(model, selected(domain), adapter_name=domain,
                                               is_trainable=False)
        else:
            model.load_adapter(selected(domain), adapter_name=domain, is_trainable=False)
    model.eval()
    features, labels = [], []
    for row in rows:
        if row['id'] not in split['fit']:
            continue
        for domain in DOMAINS:
            windows = diagnostic(row, domain)
            previous = windows.get(0)
            current = windows.get(32)
            if previous is None or current is None:
                continue
            features.append(feature(row, domain, 32, previous['a']))
            labels.append(current['a'])
    vec = DictVectorizer()
    X = vec.fit_transform(features)
    policy = LogisticRegression(max_iter=600, C=1.0, random_state=42).fit(X, labels)
    assert set(policy.classes_) == set(range(9))
    return model, vec, policy


@torch.inference_mode()
def probe(model, row, vec, policy):
    domain = row['domain']; model.set_adapter(domain)
    windows = diagnostic(row, domain)
    if 0 not in windows or 32 not in windows:
        return {'state': 'no_qualified_position32'}
    j = int(policy.predict(vec.transform([feature(row, domain, 32, windows[0]['a'])]))[0])
    response = row['response_ids']
    context = row['prompt_ids'] + response[:32]
    c = CachedModel(model)
    proposed = []
    top2 = []
    sync(); start = time.perf_counter()
    for _ in range(8):
        logits = c.next(context + proposed)
        # torch.topk may choose a different ID than argmax on exact BF16 ties.
        # The production decoder uses argmax, so preserve its token exactly.
        greedy = int(logits.argmax().item())
        alternative = int(logits.topk(3).indices.tolist()[0])
        if alternative == greedy:
            alternative = next(int(t) for t in logits.topk(3).indices.tolist() if int(t) != greedy)
        top2.append([greedy, alternative])
        proposed.append(greedy)
    sync(); main_ms = (time.perf_counter()-start)*1000
    assert proposed == windows[32]['proposal'], 'Draft replay mismatch; no score committed'
    a = windows[32]['a']
    result = {'state': 'complete', 'id': row['id'], 'domain': domain,
              'position': 32, 'predicted_rejection_position': j,
              'actual_accepted': a, 'predicted_exact': j == a,
              'draft_main_ms': main_ms, 'proposal': proposed,
              'target_correction': response[32+a] if a < 8 else None,
              'top2_at_prediction': top2[j] if j < 8 else None,
              'branch_generated': False, 'root_hit': False,
              'reused_suffix_tokens': 0, 'branch_ms': 0.0}
    if j >= 8:
        return result
    # Branch root is the draft's runner-up at the *predicted* rejection index.
    # It is chosen without seeing the target correction. New cache is required
    # in this reference implementation; the measured cost includes its prefill.
    alternate_root = top2[j][1]
    prefix = context + proposed[:j] + [alternate_root]
    bc = CachedModel(model)
    suffix = []
    sync(); start = time.perf_counter()
    for _ in range(min(3, 7-j)):
        token = int(bc.next(prefix + suffix).argmax().item())
        suffix.append(token)
    sync(); result['branch_ms'] = (time.perf_counter()-start)*1000
    result['branch_generated'] = True
    result['branch_root'] = alternate_root
    result['branch_suffix'] = suffix
    result['root_hit'] = bool(j == a and alternate_root == response[32+a])
    if result['root_hit']:
        for token, truth in zip(suffix, response[32+a+1:]):
            if token != truth: break
            result['reused_suffix_tokens'] += 1
    return result


def main():
    assert_spec(); torch.set_num_threads(4)
    rows = records('mix16', 'router_train')
    split = split_ids(rows)
    ordered = sorted((r for r in rows if r['id'] in split['holdout']),
                     key=lambda r: hashlib.sha256((SEED+'|branch|'+r['id']).encode()).hexdigest())
    # Pre-registered cap: first 15 requests per task by the fixed hash. Retain
    # unqualified and failed rows in the denominator/status rather than filter.
    selected = []
    for domain in DOMAINS:
        selected.extend([r for r in ordered if r['domain'] == domain][:15])
    assert len(selected) == 75 and len({r['id'] for r in selected}) == 75
    binding = {'at': now(), 'ids': [r['id'] for r in selected], 'split_seed': SEED,
               'source_pilot_sha256': digest(V2 / 'data/processed/pilot_manifest.jsonl'),
               'script_sha256': digest(__file__), 'policy': 'domain expert, position32, fixed top2 runner-up, suffix3'}
    manifest = V3 / 'status/branch_input_v2.json'
    if manifest.exists():
        old = json.loads(manifest.read_text()); binding['at'] = old['at']; assert old == binding
    else:
        atomic(manifest, binding)
    model, vec, policy = model_and_policy(rows, split)
    completed = 0
    for row in selected:
        path = OUT / (hashlib.sha256(row['id'].encode()).hexdigest()+'.json')
        if path.exists():
            saved = json.loads(path.read_text()); assert saved['id'] == row['id']
        else:
            saved = probe(model, row, vec, policy)
            saved.setdefault('id', row['id']); saved.setdefault('domain', row['domain'])
            atomic(path, saved)
        completed += 1
        atomic(V3 / 'status/branch_probe_v2.json', {'state': 'running', 'at': now(),
               'pid': os.getpid(), 'completed': completed, 'total': 75,
               'script_sha256': digest(__file__)})
    items = [json.loads((OUT / (hashlib.sha256(r['id'].encode()).hexdigest()+'.json')).read_text())
             for r in selected]
    qualified = [x for x in items if x['state'] == 'complete']
    branched = [x for x in qualified if x['branch_generated']]
    result = {'at': now(), 'state': 'complete', 'scope': 'same-target-prefix, one predicted top2 correction branch; no actual async or WAN',
              'requested': 75, 'qualified': len(qualified), 'unqualified': len(items)-len(qualified),
              'actual_rejections': sum(x['actual_accepted'] < 8 for x in qualified),
              'predicted_exact': sum(x['predicted_exact'] for x in qualified),
              'branches_generated': len(branched), 'root_hits': sum(x['root_hit'] for x in branched),
              'reused_suffix_tokens': sum(x['reused_suffix_tokens'] for x in branched),
              'sum_main_draft_ms': sum(x['draft_main_ms'] for x in qualified),
              'sum_branch_ms_with_fresh_prefill': sum(x['branch_ms'] for x in branched),
              'script_sha256': digest(__file__), 'input_manifest_sha256': digest(manifest),
              'caveat': 'Counterfactual prefix replay only. Branches are discarded unless both rejection position and correction root match. Actual online correctness/speed untested.'}
    atomic(V3 / 'results/branch_probe_v2_summary.json', result)
    atomic(V3 / 'status/branch_probe_v2.json', {'state': 'complete', 'at': now(),
           'completed': completed, 'total': 75, 'summary_sha256': digest(V3 / 'results/branch_probe_v2_summary.json')})
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

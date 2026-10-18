#!/root/miniconda3/envs/spec/bin/python
"""V3 first-pass rejection/branch proxy on frozen V2 router_train diagnostics.

This is a public-prefix counterfactual, not an online or WAN speed experiment.
The held-out split is by request, never by window or expert.
"""

import collections
import hashlib
import json
import math
import pathlib
import statistics
import sys

ROOT = pathlib.Path('/root/autodl-tmp/sd-lora-preexp')
V2 = ROOT / 'experiments/v2-routing'
V3 = ROOT / 'experiments/v3-async-budget'
sys.path[:0] = [str(ROOT), str(V2 / 'scripts')]

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from spec_research.common import assert_spec, atomic, digest, now
from train import DOMAINS, records

METHODS = tuple(DOMAINS)
POSITIONS = (0, 32, 64, 128, 192)
SEED = 'v3-reject-split-20261018'


def split_ids(rows):
    result = {'fit': set(), 'calibration': set(), 'holdout': set()}
    for domain in DOMAINS:
        group = sorted((r for r in rows if r['domain'] == domain),
                       key=lambda r: hashlib.sha256((SEED + '|' + r['id']).encode()).hexdigest())
        assert len(group) == 96
        for stage, block in [('fit', group[:58]), ('calibration', group[58:77]),
                             ('holdout', group[77:])]:
            result[stage].update(r['id'] for r in block)
    assert all(len(result[s]) == n for s, n in [('fit', 290), ('calibration', 95),
                                                  ('holdout', 95)])
    assert not (result['fit'] & result['calibration'] or result['fit'] & result['holdout']
                or result['calibration'] & result['holdout'])
    return result


def diagnostic(row, method):
    name = hashlib.sha256(row['id'].encode()).hexdigest() + '.json'
    path = V2 / 'results/routing/diagnostics' / method / name
    item = json.loads(path.read_text())
    assert item['id'] == row['id'] and item['method'] == method
    for w in item['diagnostic']['windows']:
        assert w['position'] in POSITIONS and len(w['proposal']) == 8
        ref = row['response_ids'][w['position']:w['position']+8]
        expected = next((k for k, (x, y) in enumerate(zip(w['proposal'], ref)) if x != y), 8)
        assert w['a'] == expected and len(ref) == 8
    return {w['position']: w for w in item['diagnostic']['windows']}


def feature(row, expert, position, previous_a):
    return {'domain=' + row['domain']: 1, 'expert=' + expert: 1,
            'position=' + str(position): 1, 'previous_a': float(previous_a),
            'previous_full': int(previous_a == 8),
            'position_tokens': position / 192}


def score(items, probabilities):
    labels = np.asarray([item['a'] for item in items], dtype=int)
    assert probabilities.shape == (len(items), 9)
    event = (labels < 4).astype(float)
    predicted = probabilities[:, :4].sum(axis=1)
    return {'n_windows': len(items), 'n_requests': len(set(item['id'] for item in items)),
            'early_reject_rate': float(event.mean()),
            'brier_early4': float(np.mean((predicted-event)**2)),
            'logloss_position': float(-np.mean(np.log(np.clip(probabilities[np.arange(len(items)),labels], 1e-9, 1)))),
            'mae_expected_a': float(np.mean(np.abs(probabilities @ np.arange(9) - labels)))}


def main():
    assert_spec()
    rows = records('mix16', 'router_train')
    split = split_ids(rows)
    per_request = {}
    items = []
    for row in rows:
        all_diag = {expert: diagnostic(row, expert) for expert in METHODS}
        per_request[row['id']] = {'row': row, 'diagnostics': all_diag}
        for expert, windows in all_diag.items():
            previous_a = None
            for position in POSITIONS:
                if position not in windows:
                    continue
                w = windows[position]
                # Previous public-prefix window is a proxy for verified history;
                # it is not a contiguous online trajectory.
                if previous_a is not None:
                    items.append({'id': row['id'], 'domain': row['domain'], 'expert': expert,
                                  'position': position, 'previous_a': previous_a,
                                  'a': w['a'], 'features': feature(row, expert, position, previous_a)})
                previous_a = w['a']
    groups = {name: [x for x in items if x['id'] in ids] for name, ids in split.items()}
    fit = groups['fit']; held = groups['holdout']
    assert fit and held
    counts = np.bincount([x['a'] for x in fit], minlength=9)
    marginal = (counts + 1) / (len(fit) + 9)
    def historical(x):
        selected = [a['a'] for a in fit if (a['previous_a'] >= 4) == (x['previous_a'] >= 4)]
        c = np.bincount(selected, minlength=9)
        return (c + 1) / (len(selected) + 9)
    vec = DictVectorizer()
    x_fit = vec.fit_transform([x['features'] for x in fit])
    model = LogisticRegression(max_iter=600, C=1.0, random_state=42)
    model.fit(x_fit, [x['a'] for x in fit])
    raw = model.predict_proba(vec.transform([x['features'] for x in held]))
    probabilities = np.zeros((len(held), 9))
    probabilities[:, model.classes_] = raw
    baseline = np.tile(marginal, (len(held), 1))
    history = np.asarray([historical(x) for x in held])
    scores = {'marginal': score(held, baseline), 'history_bucket': score(held, history),
              'domain_expert_history': score(held, probabilities)}
    # Paired request bootstrap on Brier improvement over strongest simple baseline.
    strongest = min(('marginal', 'history_bucket'), key=lambda x: scores[x]['brier_early4'])
    base_probs = baseline if strongest == 'marginal' else history
    targets = np.asarray([x['a'] < 4 for x in held], dtype=float)
    diff = (base_probs[:, :4].sum(axis=1)-targets)**2 - (probabilities[:, :4].sum(axis=1)-targets)**2
    # A short target response can have no qualified public-prefix window.
    # Keep the request in the manifest and report coverage, but bootstrap only
    # requests that produced at least one observable label.
    req_diff = {rid: float(np.mean([d for x,d in zip(held,diff) if x['id'] == rid]))
                for rid in {x['id'] for x in held}}
    rng = np.random.default_rng(42)
    values = np.asarray(list(req_diff.values()))
    bootstrap = [float(np.mean(rng.choice(values, size=len(values), replace=True))) for _ in range(2000)]
    interval = np.quantile(bootstrap, [.025, .975]).tolist()
    branch = collections.Counter()
    # Branch proxy: rank one alternate expert by FIT-only task prior. Its full
    # proposal is charged, even when the branch is never used.
    priors = {(domain, expert): statistics.mean(x['a'] for x in fit
              if x['domain'] == domain and x['expert'] == expert)
              for domain in DOMAINS for expert in METHODS}
    held_ids = split['holdout']
    for rid in held_ids:
        record = per_request[rid]
        row = record['row']; current = row['domain']
        alternate = max((e for e in METHODS if e != current),
                        key=lambda e: (priors[(current,e)], -METHODS.index(e)))
        for position, w in record['diagnostics'][current].items():
            alt = record['diagnostics'][alternate].get(position)
            if alt is None:
                continue
            branch['windows'] += 1
            if w['a'] == 8:
                branch['no_rejection'] += 1
                continue
            branch['rejected'] += 1
            j = w['a']; ref = row['response_ids'][position:]
            if alt['proposal'][:j] == ref[:j] and alt['proposal'][j] == ref[j]:
                branch['root_hit_if_parallel_generated'] += 1
                # Only suffix after corrected token can be useful. Count exact
                # following tokens, before requiring any additional verification.
                useful = 0
                for x,y in zip(alt['proposal'][j+1:], ref[j+1:]):
                    if x != y: break
                    useful += 1
                branch['potential_suffix_tokens'] += useful
    result = {'at': now(), 'scope': 'offline public-prefix proxy; not online/WAN speed',
              'state': 'complete', 'split_seed': SEED,
              'split_ids_sha256': {stage: hashlib.sha256('\n'.join(sorted(ids)).encode()).hexdigest()
                                   for stage, ids in split.items()},
              'source_manifest_sha256': digest(V2 / 'data/processed/pilot_manifest.jsonl'),
              'source_script_sha256': digest(__file__),
              'counts': {stage: {'requests': len(ids), 'windows': len(groups[stage])}
                         for stage, ids in split.items()},
              'scores': scores, 'strongest_simple_baseline': strongest,
              'heldout_brier_improvement': float(np.mean(values)),
              'heldout_brier_improvement_ci95': interval,
              'branch_proxy': dict(branch),
              'branch_proxy_caveat': 'Alternate proposal was generated at same public target prefix; selection ignores online KV and charges a full extra L8 proposal. No claim of async or net speed.'}
    atomic(V3 / 'results/offline_probe.json', result)
    atomic(V3 / 'status/offline_probe.json', {'state': 'complete', 'at': now(),
           'script_sha256': digest(__file__), 'result_sha256': digest(V3 / 'results/offline_probe.json')})
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

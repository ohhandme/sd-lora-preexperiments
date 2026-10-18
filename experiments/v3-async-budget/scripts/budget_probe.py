#!/root/miniconda3/envs/spec/bin/python
"""Prompt-only, feedback-updated expert choice on V2 public-prefix windows.

Only one expert is evaluated per decision. Counterfactual labels are read after
the action for scoring; they never enter the policy state or online reward.
"""

import collections
import hashlib
import json
import pathlib
import statistics
import sys

ROOT = pathlib.Path('/root/autodl-tmp/sd-lora-preexp')
V2 = ROOT / 'experiments/v2-routing'
V3 = ROOT / 'experiments/v3-async-budget'
sys.path[:0] = [str(ROOT), str(V2 / 'scripts'), str(V3 / 'scripts')]

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from spec_research.common import assert_spec, atomic, digest, now
from train import DOMAINS, records
from offline_probe import diagnostic, split_ids


def request_data(rows, splits):
    out = {}
    for row in rows:
        mix = {}
        if row['id'] in splits['calibration'] | splits['holdout']:
            path = V3/'results/mix80_router_train'/(hashlib.sha256(row['id'].encode()).hexdigest()+'.json')
            item = json.loads(path.read_text())
            assert item['id'] == row['id'] and item['teacher_sha256'] == row['record_sha']
            mix = {w['position']:w for w in item['diagnostic']['windows']}
        out[row['id']] = {'row': row,
                          'windows': {e: diagnostic(row, e) for e in DOMAINS},
                          'mix80': mix}
    return out


def prompt_map():
    return {x['record_id']: x['prompt'] for x in
            (json.loads(line) for line in (V2/'data/processed/pilot_manifest.jsonl').read_text().splitlines()
             if line)}


def play(record, predicted_domain, prior, alpha, exploration, switch_penalty):
    expert = max(DOMAINS, key=lambda e: (prior[(predicted_domain,e)], -DOMAINS.index(e)))
    seen = collections.Counter(); reward = collections.Counter()
    chosen = []; switches = 0
    positions = [j for j in (0,32,64,128,192) if j in record['mix80']
                 and all(j in record['windows'][e] for e in DOMAINS)]
    for step, position in enumerate(positions):
        if step:
            def utility(e):
                posterior = (alpha*prior[(predicted_domain,e)] + reward[e])/(alpha+seen[e])
                bonus = exploration/(1+seen[e])**0.5
                return posterior + bonus - (switch_penalty if e != expert else 0)
            new_expert = max(DOMAINS, key=lambda e: (utility(e), -DOMAINS.index(e)))
            if new_expert != expert: switches += 1
            expert = new_expert
        a = record['windows'][expert][position]['a']
        chosen.append({'position':position, 'expert':expert, 'accepted':a,
                       'mix80':record['mix80'][position]['a'],
                       'oracle':max(record['windows'][e][position]['a'] for e in DOMAINS)})
        seen[expert] += 1; reward[expert] += a
    return {'windows': chosen, 'switches':switches, 'predicted_domain':predicted_domain}


def score(plays):
    all_windows = [w for p in plays.values() for w in p['windows']]
    assert all_windows
    per_request = [statistics.mean(w['accepted']-w['mix80'] for w in p['windows'])
                   for p in plays.values() if p['windows']]
    return {'requests': len(plays), 'qualified_requests': len(per_request),
            'windows': len(all_windows), 'chosen_mean_a':statistics.mean(w['accepted'] for w in all_windows),
            'mix80_mean_a':statistics.mean(w['mix80'] for w in all_windows),
            'oracle_mean_a':statistics.mean(w['oracle'] for w in all_windows),
            'chosen_minus_mix80_request_mean':statistics.mean(per_request),
            'switches':sum(p['switches'] for p in plays.values())}


def main():
    assert_spec()
    rows = records('mix16', 'router_train'); splits = split_ids(rows)
    data = request_data(rows, splits); prompts = prompt_map()
    fit_rows = [r for r in rows if r['id'] in splits['fit']]
    tfidf = TfidfVectorizer(ngram_range=(1,2), max_features=20000, min_df=2,
                            sublinear_tf=True)
    X = tfidf.fit_transform([prompts[r['id']] for r in fit_rows])
    classifier = LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced',random_state=42)
    classifier.fit(X, [r['domain'] for r in fit_rows])
    predicted = {r['id']: str(classifier.predict(tfidf.transform([prompts[r['id']]]))[0])
                 for r in rows if r['id'] not in splits['fit']}
    priors = {}
    for domain in DOMAINS:
        requests = [data[r['id']] for r in fit_rows if r['domain'] == domain]
        for expert in DOMAINS:
            values = [w['a'] for request in requests for w in request['windows'][expert].values()]
            priors[(domain,expert)] = statistics.mean(values)
    def evaluate(ids, config):
        return {rid:play(data[rid],predicted[rid],priors,*config) for rid in sorted(ids)}
    menu = [(a, ex, sp) for a in (2.0,8.0) for ex in (0.0,0.5)
            for sp in (0.0,0.25,0.5)]
    calibration = [(score(evaluate(splits['calibration'], c)),c) for c in menu]
    chosen_score, config = max(calibration, key=lambda t:
                               (t[0]['chosen_minus_mix80_request_mean']-t[0]['switches']*0.05/95,
                                -menu.index(t[1])))
    hold_plays = evaluate(splits['holdout'], config)
    held = score(hold_plays)
    paired = np.asarray([statistics.mean(w['accepted']-w['mix80'] for w in p['windows'])
                         for p in hold_plays.values() if p['windows']])
    rng = np.random.default_rng(42)
    ci = np.quantile([np.mean(rng.choice(paired,len(paired),replace=True))
                      for _ in range(2000)], [.025,.975]).tolist()
    result = {'at':now(), 'state':'complete', 'scope':'offline public-prefix proxy, not online speed',
              'source_manifest_sha256':digest(V2/'data/processed/pilot_manifest.jsonl'),
              'script_sha256':digest(__file__), 'fit_requests':len(splits['fit']),
              'calibration_requests':len(splits['calibration']),
              'holdout_requests':len(splits['holdout']),
              'prompt_domain_accuracy_holdout':sum(predicted[rid]==data[rid]['row']['domain']
                   for rid in splits['holdout'])/len(splits['holdout']),
              'tuning_menu':menu, 'chosen_config':config,
              'calibration_score':chosen_score, 'holdout_score':held,
              'holdout_delta_ci95':ci,
              'caveat':'Public teacher prefixes are 32 tokens apart, unlike consecutive online rounds. Exact-token online output and net latency remain untested.'}
    atomic(V3/'results/budget_probe.json',result)
    atomic(V3/'status/budget_probe.json',{'state':'complete','at':now(),
           'result_sha256':digest(V3/'results/budget_probe.json')})
    print(json.dumps(result,indent=2))


if __name__ == '__main__':main()

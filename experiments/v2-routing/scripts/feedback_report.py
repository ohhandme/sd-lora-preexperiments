#!/root/miniconda3/envs/spec/bin/python
"""Paired v2 feedback comparison; offline Oracle has no online timing."""

import hashlib
import json
import math
import pathlib
import random
import statistics
import sys

ROOT = pathlib.Path('/root/autodl-tmp/sd-lora-preexp')
V2 = ROOT / 'experiments/v2-routing'
sys.path[:0] = [str(ROOT), str(V2 / 'scripts')]
from spec_research.common import atomic, now
from online import OUT
from feedback_online import METHODS


def geo(values):
    return math.exp(statistics.mean(math.log(x) for x in values))


def ci(values, seed=20261017):
    rng = random.Random(seed)
    n = len(values)
    boots = sorted(geo([values[rng.randrange(n)] for _ in range(n)])
                   for _ in range(2000))
    return [boots[50], boots[1949]]


def item(method, record_id):
    path = OUT / method / (hashlib.sha256(record_id.encode()).hexdigest() + '.json')
    value = json.loads(path.read_text())
    assert value['id'] == record_id and value['token_eos_truncation_equal']
    return value


def main():
    static = json.loads((OUT / 'summary.json').read_text())
    assert static['state'] == 'complete' and static['independent_AR_requests'] == 40
    subset = json.loads((OUT / 'subset_manifest.json').read_text())
    ids = subset['ids']
    assert len(ids) == 40 and len(set(ids)) == 40
    result = {}
    for method in METHODS:
        status = json.loads((V2 / 'status' / f'online.{method}.json').read_text())
        assert status['state'] == 'complete' and status['completed'] == 40
        ar_ratios = []
        mix_ratios = []
        switches = []
        accepted = []
        domains = {d: [] for d in ('code','translation','summary','qa','math')}
        for rid in ids:
            base = item('AR8', rid)
            mix = item('mix80', rid)
            candidate = item(method, rid)
            assert candidate['result']['tokens'] == base['result']['tokens']
            ar_ms = base['result']['latency_ms']
            mix_ms = mix['result']['latency_ms'] + mix['routing_ms']
            cand_ms = candidate['result']['latency_ms'] + candidate['routing_ms']
            assert min(ar_ms, mix_ms, cand_ms) > 0
            ar_ratios.append(ar_ms / cand_ms)
            mix_ratios.append(mix_ms / cand_ms)
            switches.append(candidate['result']['switches'])
            accepted.append(statistics.mean(r['ar'] for r in candidate['result']['rounds']))
            domains[candidate['domain']].append(mix_ms / cand_ms)
        result[method] = {
            'paired_request_count': 40,
            'speedup_vs_AR_geomean': geo(ar_ratios),
            'speedup_vs_AR_ci95': ci(ar_ratios),
            'speedup_vs_MIX80_geomean': geo(mix_ratios),
            'speedup_vs_MIX80_ci95': ci(mix_ratios),
            'mean_switches_per_request': statistics.mean(switches),
            'mean_accepted_draft_length': statistics.mean(accepted),
            'by_domain_speedup_vs_MIX80': {d: geo(x) for d,x in domains.items()},
        }
    report = {'state': 'complete', 'at': now(),
              'scope': 'pilot40, eight per task, L8, single timing repeat',
              'methods': result, 'static_report': str(OUT / 'summary.json'),
              'independent_AR_requests_reused': 40,
              'additional_SD_comparisons': 80,
              'limitations': [
                  'Meta-UCB is a UCB reward baseline with eight-switch budget inspired by MetaSD, not its paper implementation.',
                  'COST-UCB uses train-only task priors and observed verification reward; no future target tokens.',
                  'One GPU and one timing repeat; no WAN claim.',
                  'Offline Oracle has no online timing.',
              ]}
    atomic(OUT / 'feedback_summary.json', report)
    print(json.dumps(report, sort_keys=True))


if __name__ == '__main__':
    main()

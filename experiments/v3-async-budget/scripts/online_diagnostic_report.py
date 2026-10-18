#!/root/miniconda3/envs/spec/bin/python
"""Audit a paired 15-request GPU online diagnostic without changing the P2 gate."""

import hashlib
import json
import math
import pathlib
import random
import statistics
import sys

ROOT = pathlib.Path('/root/autodl-tmp/sd-lora-preexp')
sys.path.insert(0, str(ROOT))
from spec_research.common import atomic, now

OUT = ROOT / 'experiments/v3-async-budget/results/online_diagnostic'
METHODS = ('AR', 'L8', 'H4-8')


def geometric_mean(values):
    return math.exp(statistics.mean(math.log(value) for value in values))


def main():
    manifest = json.loads((OUT / 'subset_manifest.json').read_text())
    ids = manifest['ids']
    assert len(ids) == 15 and len(set(ids)) == 15
    records = {method: [] for method in METHODS}
    for record_id in ids:
        key = hashlib.sha256(record_id.encode()).hexdigest() + '.json'
        pair = {method: json.loads((OUT / method / key).read_text()) for method in METHODS}
        for method in METHODS:
            item = pair[method]
            assert item['id'] == record_id and item['method'] == method
            assert item['token_eos_truncation_equal'] is True
            assert item['result']['tokens'] == pair['AR']['result']['tokens']
            assert item['eos'] == pair['AR']['eos']
            assert item['truncated'] == pair['AR']['truncated']
            records[method].append(item)
    metrics = {}
    for method in METHODS:
        data = records[method]
        ratios = [records['AR'][i]['result']['latency_ms'] / data[i]['result']['latency_ms']
                  for i in range(len(ids))]
        metrics[method] = {
            'request_count': len(data),
            'mean_latency_ms': statistics.mean(x['result']['latency_ms'] for x in data),
            'median_latency_ms': statistics.median(x['result']['latency_ms'] for x in data),
            'geo_speed_vs_ar': geometric_mean(ratios),
            'mean_target_calls': statistics.mean(x['result']['target_calls'] for x in data),
            'mean_accepted_per_round': (
                statistics.mean(r['ar'] for x in data for r in x['result']['rounds'])
                if method != 'AR' else None),
            'mean_proposed_length': (
                statistics.mean(r['Lr'] for x in data for r in x['result']['rounds'])
                if method != 'AR' else None),
            'peak_allocated_gib': max(x['peak_allocated'] for x in data) / (1024 ** 3),
        }
        if method != 'AR':
            rng = random.Random(20261018)
            boot = []
            for _ in range(5000):
                draws = [rng.randrange(len(ratios)) for _ in ratios]
                boot.append(geometric_mean([ratios[i] for i in draws]))
            boot.sort()
            metrics[method]['geo_speed_vs_ar_bootstrap95_request'] = [boot[125], boot[4874]]
    paired = [records['L8'][i]['result']['latency_ms'] /
              records['H4-8'][i]['result']['latency_ms'] for i in range(len(ids))]
    rng = random.Random(20261019)
    paired_boot = sorted(geometric_mean([paired[rng.randrange(len(paired))]
                                         for _ in paired]) for _ in range(5000))
    metrics['H4-8']['geo_speed_vs_L8'] = geometric_mean(paired)
    metrics['H4-8']['geo_speed_vs_L8_bootstrap95_request'] = [paired_boot[125], paired_boot[4874]]
    summary = {'at_utc': now(), 'state': 'complete', 'scope': 'post-P2-gate diagnostic only',
               'device': 'NVIDIA RTX 6000D', 'methods': metrics,
               'exact_token_eos_truncation_match': True,
               'request_ids': ids,
               'limits': '15 router_train requests, one timing repeat, no asynchronous branch or WAN; not formal v3 P3'}
    atomic(OUT / 'summary.json', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

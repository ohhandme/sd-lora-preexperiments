#!/root/miniconda3/envs/spec/bin/python
"""Summarize complete paired v2 pilot latencies without treating Oracle as online."""

import hashlib
import json
import math
import pathlib
import random
import statistics
import sys

ROOT = pathlib.Path('/root/autodl-tmp/sd-lora-preexp')
V2 = ROOT/'experiments/v2-routing'
sys.path[:0] = [str(ROOT), str(V2/'scripts')]
from spec_research.common import atomic, now
from online import METHODS, OUT


def geometric(values):
    return math.exp(statistics.mean(math.log(v) for v in values))


def main():
    subset = json.loads((OUT/'subset_manifest.json').read_text())
    ids = subset['ids']
    assert len(ids) == 40 and len(set(ids)) == 40
    indexed = {}
    for method in METHODS:
        status = json.loads((V2/'status'/f'online.{method}.json').read_text())
        assert status['state'] == 'complete' and status['completed'] == 40
        for record_id in ids:
            path = OUT/method/(hashlib.sha256(record_id.encode()).hexdigest()+'.json')
            item = json.loads(path.read_text())
            assert item['id'] == record_id and item['token_eos_truncation_equal']
            indexed[(record_id, method)] = item
    comparison = {}
    base = [indexed[(rid,'AR8')] for rid in ids]
    for method in METHODS:
        if method == 'AR8':
            continue
        ratios = []
        requested = []
        candidate_seconds = 0.0
        baseline_seconds = 0.0
        for rid in ids:
            ar = indexed[(rid,'AR8')]
            sd = indexed[(rid,method)]
            assert sd['result']['tokens'] == ar['result']['tokens']
            actual_latency = sd['result']['latency_ms'] + sd['routing_ms']
            assert actual_latency > 0 and ar['result']['latency_ms'] > 0
            ratios.append(ar['result']['latency_ms']/actual_latency)
            requested.append({'id':rid, 'domain': ar['domain'],
                              'ar_ms':ar['result']['latency_ms'],
                              'sd_ms_including_route':actual_latency,
                              'speedup':ratios[-1],
                              'accepted_per_round':statistics.mean(x['ar'] for x in sd['result']['rounds']),
                              'sd_peak_allocated':sd['peak_allocated']})
            candidate_seconds += actual_latency/1000
            baseline_seconds += ar['result']['latency_ms']/1000
        rng = random.Random(20261017)
        boots = []
        for _ in range(2000):
            sample = [ratios[rng.randrange(40)] for _ in range(40)]
            boots.append(geometric(sample))
        boots.sort()
        comparison[method] = {'paired_request_count':40,
                              'speedup_geomean':geometric(ratios),
                              'speedup_ci95_request_bootstrap':
                              [boots[50],boots[1949]],
                              'total_ar_seconds':baseline_seconds,
                              'total_sd_including_route_seconds':candidate_seconds,
                              'peak_allocated_bytes_max':max(x['sd_peak_allocated'] for x in requested),
                              'mean_accepted_per_round':statistics.mean(x['accepted_per_round'] for x in requested),
                              'by_domain':{d:geometric([x['speedup'] for x in requested if x['domain']==d])
                                           for d in ('code','translation','summary','qa','math')}}
    report = {'state':'complete','at':now(),'scope':'pilot40, eight per task, L8, single timing repeat',
              'target':'Qwen3-8B BF16 greedy non-thinking', 'draft':'Qwen3-0.6B BF16',
              'independent_AR_requests':len(base),'SD_comparisons':40*(len(METHODS)-1),
              'methods':comparison,'limitations':
              ['Single repeat does not quantify run-to-run latency variability.',
               'One GPU only; not a WAN/edge-cloud throughput measurement.',
               'Oracle is offline and has no online timing.',
               'Existing Python cached decoder may have implementation overhead.']}
    atomic(OUT/'summary.json',report)
    print(json.dumps(report,sort_keys=True))


if __name__ == '__main__':
    main()

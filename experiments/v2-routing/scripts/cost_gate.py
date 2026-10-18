#!/root/miniconda3/envs/spec/bin/python
"""Isolated P2 hardware-cost screen; not an online Oracle measurement."""

import json
import pathlib
import statistics
import sys
import time

ROOT = pathlib.Path('/root/autodl-tmp/sd-lora-preexp')
V2 = ROOT / 'experiments/v2-routing'
sys.path[:0] = [str(ROOT), str(V2 / 'scripts')]

import torch
from spec_research.common import assert_spec, atomic, load_model, now
from spec_research.decoding import CachedModel
from train import DOMAINS, METHODS, records
from quality import pool


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def main():
    assert_spec()
    torch.set_num_threads(4)
    quality = json.loads((V2 / 'results/quality/summary.json').read_text())
    routing = json.loads((V2 / 'results/routing/summary.json').read_text())
    assert quality['state'] == routing['state'] == 'complete'
    assert quality['completed_entries'] == 1920
    assert routing['qualified_val'] == 120
    assert quality['quality_headroom_only']

    selected = {m: json.loads((V2 / 'adapters/0.6B' / m / 'seed42/selected.json').read_text())
                for m in METHODS}
    draft = pool(selected)
    for method in DOMAINS:
        draft.set_adapter(method)
    torch.cuda.synchronize()
    switches = []
    names = list(DOMAINS) * 24
    for index, method in enumerate(names):
        torch.cuda.synchronize()
        start = time.perf_counter()
        draft.set_adapter(method)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        if index >= 20:
            switches.append(elapsed)
    del draft
    torch.cuda.empty_cache()

    target = load_model('8B').eval()
    rows = [row for domain in DOMAINS
            for row in [r for r in records(domain, 'expert_train') if len(r['response_ids']) >= 8][:4]]
    assert len(rows) >= 15
    verify_ms = []
    with torch.inference_mode():
        for row in rows:
            cache = CachedModel(target)
            cache.feed(row['prompt_ids'])
            torch.cuda.synchronize()
            start = time.perf_counter()
            cache.feed(row['response_ids'][:8])
            torch.cuda.synchronize()
            verify_ms.append((time.perf_counter() - start) * 1000)

    mix = quality['means']['mix80']['macro']
    oracle = quality['means']['ORACLE']['macro']
    rounds_saved_256 = 256 / (1 + mix) - 256 / (1 + oracle)
    one_request_overhead_ms = (routing['router_ms']['acceptance'] +
                               statistics.median(switches))
    # A retrospective Oracle can only set an optimistic ceiling. The formula
    # ignores draft calls, EOS, cache/switch effects, and router prediction loss.
    optimistic_verify_saving_ms = rounds_saved_256 * statistics.median(verify_ms)
    result = {
        'state': 'complete', 'at': now(), 'partition_for_timing': 'expert_train',
        'device': torch.cuda.get_device_name(), 'request_length_assumption': 256,
        'L': 8, 'switch_samples': len(switches), 'verify_samples': len(verify_ms),
        'adapter_switch_ms_median': statistics.median(switches),
        'adapter_switch_ms_p95': percentile(switches, .95),
        'target_verify_8_ms_median': statistics.median(verify_ms),
        'target_verify_8_ms_p95': percentile(verify_ms, .95),
        'prompt_router_ms_mean': routing['router_ms']['acceptance'],
        'offline_oracle_mean_a': oracle, 'offline_mix80_mean_a': mix,
        'optimistic_rounds_saved_256': rounds_saved_256,
        'optimistic_verify_saving_ms': optimistic_verify_saving_ms,
        'single_request_route_and_switch_ms': one_request_overhead_ms,
        'optimistic_hardware_screen': optimistic_verify_saving_ms > one_request_overhead_ms,
        'observed_prompt_route_minus_mix80_a': routing['mean_a']['acc_score'] -
                                                routing['mean_a']['mix80'],
        'gate_interpretation': 'Upper-bound hardware plausibility only. Retrospective Oracle is not a feasible router. A deployable method must still beat strong baselines in exact-token online tests.',
    }
    atomic(V2 / 'results/cost_gate/summary.json', result)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()

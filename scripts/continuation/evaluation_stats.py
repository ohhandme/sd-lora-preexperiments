"""Request-level statistics and strict, auditable JSONL recovery (CPU only)."""
import collections
import hashlib
import json
import math
import pathlib
import statistics

DOMAINS = ('math', 'code', 'general')


def journal(path, recovery_dir=None):
    """Only an interrupted final JSON object may be recovered, with evidence kept."""
    path = pathlib.Path(path)
    if not path.exists():
        return
    with path.open('rb') as f:
        offset = 0
        for line in f:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                assert not line.endswith(b'\n') and not f.read(1), f'Corrupt interior record: {path}:{offset}'
                assert recovery_dir is not None, f'Interrupted journal requires explicit recovery: {path}'
                raw = path.read_bytes()
                backup = pathlib.Path(recovery_dir) / (path.name + '.' + hashlib.sha256(raw).hexdigest() + '.interrupted')
                backup.parent.mkdir(parents=True, exist_ok=True)
                backup.write_bytes(raw)
                with path.open('r+b') as out:
                    out.truncate(offset)
                return
            assert isinstance(item, dict), f'Non-object journal record: {path}'
            yield item
            offset += len(line)
            if not line.endswith(b'\n'):
                # A complete last object can have lost only its newline. Preserve
                # evidence and finish that delimiter before any future append.
                assert not f.read(1) and recovery_dir is not None
                raw = path.read_bytes()
                backup = pathlib.Path(recovery_dir)/(path.name+'.'+hashlib.sha256(raw).hexdigest()+'.missing-newline')
                backup.parent.mkdir(parents=True, exist_ok=True); backup.write_bytes(raw)
                with path.open('ab') as out: out.write(b'\n')
                return


def unique_index(items, key):
    result = {}
    for item in items:
        k = key(item)
        if k in result:
            assert result[k] == item, f'Conflicting duplicate: {k}'
        result[k] = item
    return result


def mean(values):
    values = list(values)
    return statistics.mean(values) if values else None


def percentile(values, q):
    values = sorted(values)
    if not values:
        return None
    x = (len(values)-1)*q
    lo = int(x)
    hi = min(lo+1, len(values)-1)
    return values[lo]*(hi-x) + values[hi]*(x-lo) if hi != lo else values[lo]


def paired_delta(records, candidate, reference, bootstrap=2000):
    import numpy as np
    groups = {d: [r['values'][candidate]-r['values'][reference] for r in records
                  if r['domain'] == d and r['values'][candidate] is not None and r['values'][reference] is not None]
              for d in DOMAINS}
    if not all(groups.values()):
        return dict(macro_delta=None, ci95=None, covered_requests={d: len(v) for d, v in groups.items()})
    rng = np.random.default_rng(42)
    samples = []
    for values in groups.values():
        a = np.asarray(values, dtype=float)
        samples.append(a[rng.integers(len(a), size=(bootstrap, len(a)))].mean(axis=1))
    return dict(macro_delta=mean(mean(v) for v in groups.values()),
                ci95=np.quantile(np.mean(samples, axis=0), [.025, .975]).tolist(),
                covered_requests={d: len(v) for d, v in groups.items()},
                unit='paired request, stratified by domain', bootstrap=bootstrap)


def aggregate_online(entries):
    """Ratios of sums, preserving EOS-short requests and router time."""
    assert entries
    assert len({x['id'] for x in entries}) == len(entries)
    for x in entries:
        assert x['token_match'] and x['ar']['tokens'] == x['sd']['tokens']
        assert 0 < x['ar']['latency_ms'] and 0 < x['sd']['latency_ms']
        rounds = x['sd']['rounds']
        assert sum(r['nr'] for r in rounds) == len(x['sd']['tokens'])
        assert all(0 <= r['ar'] <= r['Lr'] and r['nr'] > 0 for r in rounds)
    domains = {}
    for d in DOMAINS:
        rr = [x for x in entries if x['domain'] == d]
        if not rr:
            continue
        sums = {m: sum(x[m]['latency_ms'] for x in rr) for m in ('ar', 'sd')}
        domains[d] = dict(requests=len(rr), ar_total_ms=sums['ar'], sd_total_ms=sums['sd'],
                          speedup=sums['ar']/sums['sd'],
                          confirmed_tokens_per_s=sum(len(x['sd']['tokens']) for x in rr)*1000/sums['sd'])
    rounds = [r for x in entries for r in x['sd']['rounds']]
    accepted = [r['ar'] for r in rounds]
    L = entries[0]['L']
    after_first = [x for x in entries if len(x['sd']['tokens']) > 1]
    decode_time = sum(x['sd']['decode_seconds'] for x in after_first)
    tpot = [x['sd']['decode_seconds']*1000/(len(x['sd']['tokens'])-1) for x in after_first]
    rng_seed = 42
    import numpy as np
    rng = np.random.default_rng(rng_seed)
    domain_boot = []
    for d in DOMAINS:
        rr = [x for x in entries if x['domain'] == d]
        if not rr:
            continue
        idx = rng.integers(len(rr), size=(2000, len(rr)))
        a = np.asarray([x['ar']['latency_ms'] for x in rr]); b = np.asarray([x['sd']['latency_ms'] for x in rr])
        domain_boot.append(a[idx].sum(axis=1)/b[idx].sum(axis=1))
    macro_boot = np.exp(np.mean(np.log(domain_boot), axis=0)) if len(domain_boot) == 3 else None
    return dict(requests=len(entries), domain=domains,
                macro_speedup_geo=math.exp(mean(math.log(domains[d]['speedup']) for d in DOMAINS)) if len(domains) == 3 else None,
                macro_speedup_geo_ci95=np.quantile(macro_boot, [.025, .975]).tolist() if macro_boot is not None else None,
                bootstrap_unit='paired request, stratified by domain; timing repeat reported separately',
                confirmed_tokens_per_s=sum(len(x['sd']['tokens']) for x in entries)*1000/sum(x['sd']['latency_ms'] for x in entries),
                decode_tokens_per_s=sum(len(x['sd']['tokens'])-1 for x in after_first)/decode_time if decode_time else None,
                draft_acceptance_rate=sum(accepted)/sum(r['Lr'] for r in rounds),
                mean_accepted_draft_length=mean(accepted), mean_committed_per_round=mean(r['nr'] for r in rounds),
                accepted_length_hist=dict(collections.Counter(accepted)),
                survival=[mean(a >= k for a in accepted) for k in range(1, L+1)],
                ttft_ms=dict(median=percentile([x['sd']['ttft_ms'] for x in entries], .5), p95=percentile([x['sd']['ttft_ms'] for x in entries], .95)),
                tpot_ms=dict(median=percentile(tpot, .5), p95=percentile(tpot, .95)),
                routing_ms=dict(median=percentile([x['routing_ms'] for x in entries], .5), p95=percentile([x['routing_ms'] for x in entries], .95)),
                target_calls=sum(x['sd']['target_calls'] for x in entries),
                target_prefill_calls=sum(x['sd']['target_prefill_calls'] for x in entries),
                target_verify_calls=sum(x['sd']['target_verify_calls'] for x in entries),
                draft_calls=sum(x['sd']['draft_calls'] for x in entries), rounds=len(rounds),
                peak_allocated=max(x['peak_allocated'] for x in entries), peak_reserved=max(x['peak_reserved'] for x in entries),
                output_mismatch_rate=0., note='Single-request local draft/verify; no network simulation or online Oracle')


def online_key(row):
    return row['repeat'], row['method'], row['L'], row['id']


def paired_latency_ratio(entries, candidate, reference, repeat, L, domain_n=200):
    """Direct method comparison, matching IDs within the same timing repeat."""
    import numpy as np
    selected = [x for x in entries if x['repeat'] == repeat and x['L'] == L and x['method'] in (candidate, reference)]
    index = unique_index(selected, lambda x:(x['method'],x['id']))
    ids = sorted({x['id'] for x in selected if x['method'] == candidate})
    assert len(ids) == 3*domain_n and all((reference,i) in index for i in ids)
    rng = np.random.default_rng(42); domain = {}; boot = []
    for d in DOMAINS:
        dr = [i for i in ids if index[candidate,i]['domain'] == d]
        assert len(dr) == domain_n and all(index[reference,i]['domain'] == d for i in dr)
        a = np.asarray([index[reference,i]['sd_ms'] for i in dr]); b = np.asarray([index[candidate,i]['sd_ms'] for i in dr])
        assert np.all(a > 0) and np.all(b > 0)
        domain[d] = float(a.sum()/b.sum())
        ix = rng.integers(len(dr), size=(2000,len(dr))); boot.append(a[ix].sum(axis=1)/b[ix].sum(axis=1))
    macro = math.exp(mean(math.log(v) for v in domain.values()))
    ci = np.quantile(np.exp(np.mean(np.log(boot),axis=0)), [.025,.975]).tolist()
    return dict(candidate=candidate, reference=reference, repeat=repeat, L=L, requests=len(ids),
                domain_speedup_reference_to_candidate=domain, macro_speedup_geo=macro, ci95=ci,
                bootstrap_unit='paired request, domain-stratified, 2000 resamples')

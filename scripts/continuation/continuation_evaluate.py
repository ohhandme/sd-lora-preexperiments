"""Frozen seed42 validation, request-level routing, and actual greedy draft/verify.

Helpers live outside the P0-protected implementation. No core decoder, precision,
teacher record, model, split, training parameter, or program is modified here.
"""
import argparse
import collections
import contextlib
import gc
import json
import math
import os
import pathlib
import pickle
import random
import statistics
import subprocess
import sys
import time

ROOT = pathlib.Path('/root/autodl-tmp/sd-lora-preexp')
sys.path.insert(0, str(ROOT))
from spec_research.common import *
from spec_research.data import partition
from spec_research.teacher import validate_teacher
from spec_research.decoding import public_prefix, ar, speculative, compare, sync
from spec_research.gate import validate_gate
from continuation_worker import validate_diagnostic
from evaluation_stats import journal, unique_index, mean, paired_delta, percentile, aggregate_online, online_key, paired_latency_ratio
from peft import PeftModel
import torch

RUN = 'overnight-20261017-seed42'
BASE = ROOT/'results/continuation-20261017'
METHODS = [*DOMAINS, 'mix16', 'mix48']
ONLINE = ['BASE', 'mix16', 'mix48', 'DOMAIN', 'FIXEDBEST', 'ROUTE-TASK', 'ROUTE-ACC']
RECOVERY = BASE/'recovery_archives'


def records(parts=('adapter_val',)):
    return [r for d in DOMAINS for p in parts for r in partition(d, p)]


def preliminary_records():
    all_records = records(); rng = random.Random(20261017); chosen = set()
    for d in DOMAINS:
        chosen.update(x['id'] for x in rng.sample([r for r in all_records if r['domain'] == d], 8))
    result = [r for r in all_records if r['id'] in chosen]
    assert len(result) == 24 and all(sum(x['domain'] == d for x in result) == 8 for d in DOMAINS)
    return result


def online_out(scale, preliminary=False):
    return BASE/('preliminary_online' if preliminary else 'online')/scale


def selection(scale, method):
    row = read(ROOT/'adapters'/scale/method/'seed42'/RUN/'selected.json')
    assert row['provenance']['source_hashes'] == sources(), f'Stale selected adapter: {scale}/{method}'
    return row


def selected(scale, method):
    return selection(scale, method)['adapter_path']


def pool(scale, methods):
    model = load_model(scale)
    for i, method in enumerate(methods):
        if i == 0:
            model = PeftModel.from_pretrained(model, selected(scale, method), adapter_name=method, is_trainable=False)
        else:
            model.load_adapter(selected(scale, method), adapter_name=method, is_trainable=False)
    return model.eval()


def router_path(scale):
    return ROOT/'routers/acceptance'/scale/'seed42'/RUN/'router.pkl'


def task_path():
    return ROOT/'routers/task/seed42/overnight-20261017/router.pkl'


def classifier(path):
    with pathlib.Path(path).open('rb') as f:
        return pickle.load(f)


def bind(out, scale, tt, extra=None):
    """Bind every resumed result to the exact selected weights and frozen inputs."""
    expected = dict(source_hashes=sources(), seed=42, scale=scale, partition_ids=[r['id'] for r in tt],
                    teacher_records={r['id']: r['record_sha'] for r in tt},
                    adapters={m: digest(pathlib.Path(selected(scale, m))/'adapter_model.safetensors') for m in METHODS},
                    task_router_sha256=digest(task_path()), policy='request B1, BF16, greedy, max_new256, public positions0/32/64/128/192',
                    extra=extra or {})
    p = out/'input_manifest.json'
    if p.exists():
        assert read(p) == expected, f'Evaluation inputs changed: {p}'
    else:
        atomic(p, expected)
    return objhash(expected)


def progress(out, **kw):
    atomic(out/'progress.json', dict(at=now(), **kw))
    print(json.dumps(dict(at=now(), **kw)), flush=True)


def quality_report(scale):
    out = BASE/'quality'/scale
    indexed = unique_index(journal(out/'matrix.jsonl', RECOVERY), lambda x: (x['id'], x['method'], x['L']))
    rr = records(); task = classifier(task_path())
    acc = classifier(router_path(scale)) if router_path(scale).exists() else None
    predictions = {'ROUTE-TASK': dict(zip([r['id'] for r in rr], map(str, task.predict([r['prompt'] for r in rr]))))}
    if acc is not None:
        predictions['ROUTE-ACC'] = dict(zip([r['id'] for r in rr], map(str, acc.predict([r['prompt'] for r in rr]))))
    summaries = {}
    for L in sorted({k[2] for k in indexed}):
        complete = [r for r in rr if all((r['id'], m, L) in indexed for m in ['BASE', *METHODS])]
        qualified = [r for r in complete if indexed[r['id'], 'BASE', L]['diagnostic']['mean_a'] is not None]
        fixed_scores = {m: mean(mean(indexed[r['id'], m, L]['diagnostic']['mean_a'] for r in qualified if r['domain'] == d) for d in DOMAINS)
                        for m in DOMAINS} if all(any(r['domain'] == d for r in qualified) for d in DOMAINS) else {}
        fixed = max(DOMAINS, key=lambda m: fixed_scores[m]) if fixed_scores else None
        values = []
        chosen = {name: [] for name in predictions}
        for r in qualified:
            diag = {m: indexed[r['id'], m, L]['diagnostic'] for m in ['BASE', *METHODS]}
            assert all(x['mean_a'] is not None for x in diag.values())
            scores = {d: diag[d]['mean_a'] for d in DOMAINS}; best = max(scores.values())
            v = {m: x['mean_a'] for m, x in diag.items()}
            v.update(DOMAIN=scores[r['domain']], ORACLE=best, FIXEDBEST=scores[fixed] if fixed else None)
            for name, pred in predictions.items():
                p = pred[r['id']]; assert p in DOMAINS
                v[name] = scores[p]
                chosen[name].append(dict(id=r['id'], domain=r['domain'], selected=p, optimal_set_hit=scores[p] == best,
                                         oracle_regret=best-scores[p], domain_correct=p == r['domain']))
            values.append(dict(id=r['id'], domain=r['domain'], values=v))
        names = ['BASE', *METHODS, 'DOMAIN', 'FIXEDBEST', 'ORACLE', *predictions]
        metrics = {}
        for name in names:
            per_domain = {d: [x['values'][name] for x in values if x['domain'] == d and x['values'][name] is not None] for d in DOMAINS}
            means = {d: mean(v) for d, v in per_domain.items()}
            metrics[name] = dict(domain_mean_a=means, macro_mean_a=mean(means.values()) if all(v is not None for v in means.values()) else None,
                                 covered_requests={d: len(v) for d, v in per_domain.items()})
        paired = {f'{m}_minus_{ref}': paired_delta(values, m, ref) for m in ['DOMAIN', 'ORACLE', 'FIXEDBEST', *predictions]
                  for ref in ('mix16', 'mix48')} if values else {}
        # Survival is first averaged within a request, then across requests.
        survival = {}; hist = {}; coverage = {}
        for m in ['BASE', *METHODS]:
            survival[m] = {}
            hist[m] = {}
            for d in DOMAINS:
                dd = [indexed[r['id'], m, L]['diagnostic'] for r in qualified if r['domain'] == d]
                survival[m][d] = [mean(x['survival'][k] for x in dd) for k in range(L)]
                hist[m][d] = dict(collections.Counter(w['a'] for x in dd for w in x['windows']))
        for d in DOMAINS:
            dr = [r for r in complete if r['domain'] == d]
            coverage[d] = dict(requests=len(dr), qualified=sum(r in qualified for r in dr),
                               no_public_window=sum(indexed[r['id'], 'BASE', L]['diagnostic']['mean_a'] is None for r in dr),
                               excluded_boundary_windows=sum(5-len(indexed[r['id'], 'BASE', L]['diagnostic']['windows']) for r in dr))
        complement = {m: mean(x['values']['ORACLE']-x['values'][m] for x in values) for m in DOMAINS}
        routing = {name: dict(optimal_set_hit_rate=mean(x['optimal_set_hit'] for x in xs),
                              oracle_regret=mean(x['oracle_regret'] for x in xs),
                              source_domain_accuracy=mean(x['domain_correct'] for x in xs),
                              selection_counts=dict(collections.Counter(x['selected'] for x in xs))) for name, xs in chosen.items()}
        report = dict(state='complete' if len(complete) == 600 else 'partial', scale=scale, L=L, partition='adapter_val',
                      expected_requests=600, completed_matrix_requests=len(complete), qualified_requests=len(qualified), fixedbest=fixed,
                      fixedbest_rule='highest macro-domain request acceptance at this L; stable math/code/general ties',
                      methods=metrics, bootstrap=paired, survival=survival, accepted_length_hist=hist, coverage=coverage,
                      oracle_minus_fixed_expert_mean=complement, routing=routing,
                      note='Oracle is offline acceptance upper bound; all timings measured separately; checkpoint-selection validation, not held-out test', seed=42)
        atomic(out/f'quality_L{L}.json', report); summaries[str(L)] = report
        if L == 8:
            atomic(out/'quality_summary.json', report)
        atomic(out/f'per_request_L{L}.json', dict(partition='adapter_val', requests=values, routing=chosen))
    atomic(out/'quality_by_length.json', summaries)


def quality(scale, lengths, lock_fd=None):
    out = BASE/'quality'/scale
    if scale == '0.6B' and 8 in lengths and read(ROOT/'manifests/continuation_config.json').get('online_scope') != 'preliminary24':
        pilot = BASE/'online_pilot'/'report.json'
        if not pilot.exists():
            assert lock_fd is not None
            cmd = [SPEC+'/bin/python', '-u', str(ROOT/'scripts/continuation/online_pilot.py'), '--managed-lock-fd', str(lock_fd)]
            child = subprocess.run(cmd, cwd=ROOT, pass_fds=(lock_fd,))
            assert child.returncode == 0, 'Real-checkpoint online calibration failed; formal quality queue blocked for diagnosis'
        assert read(pilot)['state'] == 'complete'
    tt = validate_teacher(records()); refs = {r['id']: r for r in tt}
    fingerprint = bind(out, scale, tt)
    path = out/'matrix.jsonl'
    done = unique_index(journal(path, RECOVERY), lambda x: (x['id'], x['method'], x['L']))
    for (i, m, L), row in done.items():
        assert i in refs and m in ['BASE', *METHODS] and L in (4, 8, 12)
        assert row['fingerprint'] == fingerprint and row['domain'] == refs[i]['domain']
        validate_diagnostic(row['diagnostic'], refs[i]['response_ids'][:256], L)
    cached = {}
    for m in METHODS:
        epoch = selection(scale, m)['selected_epoch']
        for row in journal(ROOT/'results/validation'/f'{RUN}.{scale}.{m}.selection.jsonl', RECOVERY):
            assert row['id'] in refs
            if row['epoch'] == epoch:
                diag = validate_diagnostic(row, refs[row['id']]['response_ids'][:256])
                key = (m, row['id'])
                if key in cached: assert cached[key] == diag
                cached[key] = diag
    model = pool(scale, METHODS)
    # Replay one cached selection example per adapter/domain, on the actual model.
    audits = []
    if 8 in lengths:
        for m in METHODS:
            model.set_adapter(m)
            for d in DOMAINS:
                r = next((r for r in tt if r['domain'] == d and (m, r['id']) in cached), None)
                if r is not None:
                    assert public_prefix(model, r['prompt_ids'], r['response_ids'][:256], 8) == cached[m, r['id']]
                    audits.append(dict(method=m, domain=d, id=r['id'], token_proposals_exact=True))
        atomic(out/'selection_cache_replay.json', dict(state='passed', audits=audits, fingerprint=fingerprint))
    for L in lengths:
        for n, r in enumerate(tt):
            for m in ['BASE', *METHODS]:
                key = (r['id'], m, L)
                if key in done: continue
                if L == 8 and (m, r['id']) in cached:
                    diag = cached[m, r['id']]; origin = 'selected_epoch_cache_GPU_replay_passed'
                else:
                    cm = model.disable_adapter() if m == 'BASE' else contextlib.nullcontext()
                    if m != 'BASE': model.set_adapter(m)
                    with cm: diag = public_prefix(model, r['prompt_ids'], r['response_ids'][:256], L)
                    origin = 'autoregressive_public_prefix'
                validate_diagnostic(diag, r['response_ids'][:256], L)
                item = dict(id=r['id'], domain=r['domain'], partition='adapter_val', scale=scale, method=m, L=L,
                            diagnostic=diag, fingerprint=fingerprint, origin=origin)
                append(path, item); done[key] = item
            if n % 10 == 0:
                progress(out, job='quality', L=L, processed_requests=n+1, entries=len(done), expected_this_L=3600)
    quality_report(scale)
    progress(out, job='quality', state='complete', lengths=list(lengths), entries=len(done))


def label(scores, domain):
    best = max(scores.values()); tied = [d for d in DOMAINS if scores[d] == best]
    return domain if domain in tied else tied[0]


def acceptance_router(scale):
    out = BASE/'routing'/scale; path = out/'labels.jsonl'
    tt = validate_teacher(records(('router_train', 'router_val'))); refs = {r['id']: r for r in tt}
    fingerprint = bind(out, scale, tt)
    done = unique_index(journal(path, RECOVERY), lambda r: r['id'])
    for i, row in done.items():
        assert i in refs and row['fingerprint'] == fingerprint and row['partition'] == refs[i]['partition']
        for m in DOMAINS:
            validate_diagnostic(row['diagnostics'][m], refs[i]['response_ids'][:256])
            assert row['expert_mean_a'][m] == row['diagnostics'][m]['mean_a']
    model = pool(scale, DOMAINS)
    for n, r in enumerate(tt):
        if r['id'] in done: continue
        diags = {}
        for m in DOMAINS:
            cp = out/'expert_cache'/f'{m}.{objhash(r["id"])}.json'
            if cp.exists():
                row = read(cp); assert row['fingerprint'] == fingerprint and row['id'] == r['id']
                diag = row['diagnostic']
            else:
                model.set_adapter(m)
                diag = public_prefix(model, r['prompt_ids'], r['response_ids'][:256], 8)
                atomic(cp, dict(id=r['id'], method=m, fingerprint=fingerprint, diagnostic=diag))
            validate_diagnostic(diag, r['response_ids'][:256]); diags[m] = diag
        item = dict(id=r['id'], domain=r['domain'], partition=r['partition'], L=8, fingerprint=fingerprint,
                    expert_mean_a={m: x['mean_a'] for m, x in diags.items()},
                    coverage={m: len(x['windows']) for m, x in diags.items()}, diagnostics=diags)
        append(path, item); done[r['id']] = item
        if n % 5 == 0: progress(out, job='routing_labels', completed=len(done), expected=1500)
    del model; gc.collect(); torch.cuda.empty_cache()
    qualified = [r for r in tt if all(v is not None for v in done[r['id']]['expert_mean_a'].values())]
    train = [r for r in qualified if r['partition'] == 'router_train']; val = [r for r in qualified if r['partition'] == 'router_val']
    labels = [label(done[r['id']]['expert_mean_a'], r['domain']) for r in train]
    assert len(set(labels)) >= 2, 'Only one optimal expert class: report degeneracy before modifying classifier protocol'
    from sklearn.pipeline import make_pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    clf = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), max_features=20000, min_df=2, sublinear_tf=True),
                        LogisticRegression(C=1., max_iter=1000, class_weight='balanced', random_state=42))
    clf.fit([r['prompt'] for r in train], labels)
    dest = router_path(scale); dest.parent.mkdir(parents=True, exist_ok=True)
    temp = dest.with_suffix('.tmp')
    with temp.open('wb') as f: pickle.dump(clf, f); f.flush(); os.fsync(f.fileno())
    os.replace(temp, dest)
    task = classifier(task_path()); evaluations = []
    for r in val:
        scores = done[r['id']]['expert_mean_a']; best = max(scores.values())
        predictions = {}
        for name, c in [('ROUTE-TASK', task), ('ROUTE-ACC', clf)]:
            t = time.perf_counter(); pred = str(c.predict([r['prompt']])[0]); ms = (time.perf_counter()-t)*1000
            predictions[name] = dict(predicted=pred, label=label(scores, r['domain']), label_correct=pred == label(scores, r['domain']),
                                     optimal_set_hit=scores[pred] == best, source_domain_correct=pred == r['domain'],
                                     selected_mean_a=scores[pred], oracle_regret=best-scores[pred], feature_and_classification_ms=ms)
        evaluations.append(dict(id=r['id'], domain=r['domain'], predictions=predictions))
    aggregates = {}
    for name in ('ROUTE-TASK', 'ROUTE-ACC'):
        xs = [r['predictions'][name] for r in evaluations]
        aggregates[name] = dict(best_label_accuracy=mean(x['label_correct'] for x in xs), optimal_set_hit_rate=mean(x['optimal_set_hit'] for x in xs),
                                source_domain_accuracy=mean(x['source_domain_correct'] for x in xs), oracle_regret=mean(x['oracle_regret'] for x in xs),
                                selected_mean_a=mean(x['selected_mean_a'] for x in xs), selection_counts=dict(collections.Counter(x['predicted'] for x in xs)),
                                CPU_predict_ms=dict(median=percentile([x['feature_and_classification_ms'] for x in xs], .5), p95=percentile([x['feature_and_classification_ms'] for x in xs], .95)))
    atomic(out/'router_validation.json', dict(state='complete', fingerprint=fingerprint, classifier_sha256=digest(dest),
           train_requests=len(train), val_requests=len(val), excluded_no_public_window=len(tt)-len(qualified),
           excluded_counts=dict(collections.Counter(r['partition'] for r in tt if r not in qualified)), source_counts=dict(router_train=1200, router_val=300),
           train_label_counts=dict(collections.Counter(labels)), methods=aggregates, requests=evaluations,
           note='Prompt-only TFIDF fit on router_train; router_val separately scored; tie-optimal accuracy differs from deterministic-label accuracy; activation timed online'))
    quality_report(scale)
    progress(out, job='router', state='complete', train=len(train), val=len(val))


def resident_methods(method, fixed):
    if method == 'BASE': return []
    if method in ('mix16', 'mix48'): return [method]
    if method == 'FIXEDBEST': return [fixed]
    return list(DOMAINS)


def choose(method, r, fixed, clf):
    if method == 'BASE': return None
    if method == 'DOMAIN': return r['domain']
    if method == 'FIXEDBEST': return fixed
    if method.startswith('ROUTE-'): return str(clf.predict([r['prompt']])[0])
    return method


def weight_bytes(model):
    # Parameter storages, de-duplicated; extra numerical buffers reported separately.
    seen = set(); size = 0
    for p in model.parameters():
        k = (p.device.type, p.device.index, p.untyped_storage().data_ptr())
        if k not in seen: size += p.untyped_storage().nbytes(); seen.add(k)
    return size


def hardware():
    p = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw', '--format=csv,noheader,nounits'], capture_output=True, text=True)
    q = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'], capture_output=True, text=True)
    assert p.returncode == q.returncode == 0
    return dict(gpu=p.stdout.strip(), processes=q.stdout.strip(), measured_at=now(), note='NVML point sample, not a peak; PyTorch allocator peak separately recorded')


def check_exclusive():
    p = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], capture_output=True, text=True)
    assert p.returncode == 0
    others = [int(x) for x in p.stdout.split() if x.isdigit() and int(x) != os.getpid()]
    assert not others, f'Timing blocked by other GPU compute processes: {others}'


def ar_config(scale, L, repeat, preliminary=False):
    """AR8 is a separate, target-only configuration, shared across seven methods.

    Same request IDs, cap, precision, backend, repeat and L block. Its three
    measured repetitions are not re-counted as independent measurements when
    used by the seven paired comparisons. L blocks get fresh AR measurements.
    """
    out = online_out(scale, preliminary)
    tt = validate_teacher(preliminary_records() if preliminary else records()); expected = len(tt)
    fingerprint = bind(out, scale, tt, dict(acceptance_router_sha256=digest(router_path(scale))))
    cfgdir = out/'baselines'/f'L{L}.repeat{repeat}'
    path = cfgdir/'requests.jsonl'
    done = unique_index(journal(path, RECOVERY), lambda r:r['id'])
    check_exclusive(); start = time.perf_counter(); target = load_model('8B')
    eos = target.generation_config.eos_token_id; eos = set([eos] if isinstance(eos, int) else eos)
    atomic(cfgdir/'runtime.json', dict(state='running', configuration='AR8', resident_drafts=[], resident_adapters=[], fingerprint=fingerprint,
           startup_seconds=time.perf_counter()-start, target_weights_bytes=weight_bytes(target), pid=os.getpid(), gpu=hardware()))
    warms = [partition(d, 'expert_train')[i] for i in range(4) for d in DOMAINS][:10]
    for r in warms: ar(target, r['prompt_ids'], 256, eos)
    for n, r in enumerate(tt):
        if r['id'] in done:
            assert done[r['id']]['fingerprint'] == fingerprint and done[r['id']]['ar']['tokens'] == r['response_ids'][:256]
            continue
        if n % 10 == 0: check_exclusive()
        torch.cuda.reset_peak_memory_stats()
        result = ar(target, r['prompt_ids'], 256, eos)
        if result['tokens'] != r['response_ids'][:256]:
            atomic(cfgdir/'failure.json', dict(id=r['id'], actual=result, teacher_tokens=r['response_ids'][:256]))
            raise RuntimeError('AR/teacher trajectory changed; comparison blocked')
        row = dict(id=r['id'], domain=r['domain'], fingerprint=fingerprint, ar=result, repeat=repeat, L_block=L,
                   peak_allocated=torch.cuda.max_memory_allocated(), peak_reserved=torch.cuda.max_memory_reserved(), at=now())
        append(path, row); done[r['id']] = row
        if n % 5 == 0: progress(out, job='AR8', L=L, repeat=repeat, completed=len(done), expected=expected)
    check_exclusive(); assert len(done) == expected
    atomic(cfgdir/'complete.json', dict(state='complete', requests=expected, fingerprint=fingerprint, warmup_requests=10,
           warmup_partition='expert_train', warmup_ids=[r['id'] for r in warms], shared_by_methods=ONLINE,
           configuration_wall_seconds=time.perf_counter()-start,
           measurement_unit=f'One target-only AR8 measurement per request ({expected} requests) in this repeat/L block; reused by seven paired method comparisons',
           peak_allocated=max(x['peak_allocated'] for x in done.values()), gpu=hardware()))


def online_config(scale, method, L, repeat, lock_fd, preliminary=False):
    out = online_out(scale, preliminary); tt = validate_teacher(preliminary_records() if preliminary else records()); refs = {r['id']: r for r in tt}; expected = len(tt)
    fingerprint = bind(out, scale, tt, dict(acceptance_router_sha256=digest(router_path(scale))))
    path = out/'requests.jsonl'
    previous = unique_index((x for x in journal(path, RECOVERY) if (x['method'], x['L'], x['repeat']) == (method, L, repeat)), online_key)
    for x in previous.values():
        assert x['fingerprint'] == fingerprint and x['id'] in refs and x['token_match'] and x['ar']['tokens'] == x['sd']['tokens']
    ar_dir = out/'baselines'/f'L{L}.repeat{repeat}'
    baseline_rows = unique_index(journal(ar_dir/'requests.jsonl', RECOVERY), lambda r:r['id'])
    assert read(ar_dir/'complete.json')['fingerprint'] == fingerprint and len(baseline_rows) == expected
    cfgdir = out/'configurations'/f'L{L}.repeat{repeat}.{method}'
    # FIXEDBEST is selected once at the primary L8, avoiding re-selection on scans.
    fixed = read(BASE/'quality'/scale/'quality_summary.json')['fixedbest']; assert fixed in DOMAINS
    resident = resident_methods(method, fixed)
    clf = classifier(task_path() if method == 'ROUTE-TASK' else router_path(scale)) if method.startswith('ROUTE-') else None
    start = time.perf_counter(); check_exclusive()
    draft = pool(scale, resident)
    if resident: draft.set_adapter(resident[0])
    draft_weights = weight_bytes(draft); draft_only_allocated = torch.cuda.memory_allocated()
    target = load_model('8B'); target_weights = weight_bytes(target)
    eos = target.generation_config.eos_token_id; eos = set([eos] if isinstance(eos, int) else eos)
    check_exclusive()
    atomic(cfgdir/'runtime.json', dict(state='running', pid=os.getpid(), fingerprint=fingerprint, scale=scale, method=method, L=L, repeat=repeat,
           resident_adapters=resident, startup_seconds=time.perf_counter()-start, draft_weights_bytes=draft_weights, target_weights_bytes=target_weights,
           adapter_bytes=sum(p.numel()*p.element_size() for n, p in draft.named_parameters() if 'lora_' in n),
           draft_only_load_allocated=draft_only_allocated,
           numerical_auxiliary_buffers='Included in allocator measurements; not counted as parameter weights',
           gpu=hardware(), interpreter=sys.executable, protected_sources=sources(), helper_sha256=digest(__file__)))
    # Warmup uses expert_train, never test/validation requests; full max_new256.
    warms = [partition(d, 'expert_train')[i] for i in range(4) for d in DOMAINS][:10]
    for r in warms:
        selected_expert = choose(method, r, fixed, clf)
        if method in ('DOMAIN', 'ROUTE-TASK', 'ROUTE-ACC'): draft.set_adapter(selected_expert)
        speculative(target, draft, r['prompt_ids'], 256, eos, L)
    atomic(cfgdir/'warmup.json', dict(requests=10, partition='expert_train', ids=[r['id'] for r in warms], max_new_tokens=256, excluded_from_metrics=True))
    for n, r in enumerate(tt):
        key = (repeat, method, L, r['id'])
        if key in previous: continue
        if n % 10 == 0: check_exclusive()
        baseline_record = baseline_rows[r['id']]
        assert baseline_record['fingerprint'] == fingerprint
        baseline = baseline_record['ar']
        assert baseline['tokens'] == r['response_ids'][:256], f'AR/teacher trajectory changed: {r["id"]}'
        routing_ms = 0.; selected_expert = resident[0] if resident else None
        active_before = draft.active_adapter if resident else None
        if method in ('DOMAIN', 'ROUTE-TASK', 'ROUTE-ACC'):
            sync(); t = time.perf_counter(); selected_expert = choose(method, r, fixed, clf)
            assert selected_expert in resident
            draft.set_adapter(selected_expert); sync(); routing_ms = (time.perf_counter()-t)*1000
        torch.cuda.reset_peak_memory_stats()
        candidate = speculative(target, draft, r['prompt_ids'], 256, eos, L)
        mismatch = compare(baseline, candidate)
        if mismatch:
            atomic(cfgdir/'failure.json', dict(id=r['id'], method=method, L=L, repeat=repeat, fingerprint=fingerprint, **mismatch))
            raise RuntimeError('Full token mismatch; no speed claim or skipped request permitted')
        # Include both CPU feature extraction and expert activation in start-to-end.
        candidate['decoder_only_latency_ms'] = candidate['latency_ms']
        candidate['latency_ms'] += routing_ms; candidate['ttft_ms'] += routing_ms
        entry = dict(id=r['id'], domain=r['domain'], partition='adapter_val', scale=scale, method=method, L=L, repeat=repeat,
                     ar=baseline, sd=candidate, selected=selected_expert, routing_ms=routing_ms, expert_changed=active_before != selected_expert,
                     token_match=True, eos_match=bool(baseline['tokens'][-1] in eos) == bool(candidate['tokens'][-1] in eos),
                     truncation_match=(len(baseline['tokens']) == 256) == (len(candidate['tokens']) == 256),
                     peak_allocated=torch.cuda.max_memory_allocated(), peak_reserved=torch.cuda.max_memory_reserved(), fingerprint=fingerprint, at=now(),
                     ar_measurement_id=str(ar_dir/'requests.jsonl')+'#'+r['id'], ar_measurement_shared_across_methods=True)
        append(path, entry); previous[key] = entry
        if n % 5 == 0:
            progress(out, job='online', scale=scale, method=method, L=L, repeat=repeat, completed_config=len(previous), expected_config=expected,
                     seconds_since_load=time.perf_counter()-start)
    check_exclusive(); assert len(previous) == expected
    result = aggregate_online(list(previous.values()))
    atomic(cfgdir/'complete.json', dict(state='complete', method=method, L=L, repeat=repeat, fingerprint=fingerprint, at=now(), gpu=hardware(), **result))
    # Profiling is a separate replay, excluded from throughput/latency results.
    profile_config(target, draft, tt, previous, method, fixed, clf, eos, L, repeat, cfgdir, (.5,) if preliminary else (.1,.4,.7,.9))
    complete = read(cfgdir/'complete.json')
    complete['configuration_wall_seconds_with_profile'] = time.perf_counter()-start
    atomic(cfgdir/'complete.json', complete)


def profile_config(target, draft, tt, previous, method, fixed, clf, eos, L, repeat, out, quantiles=(.1,.4,.7,.9)):
    import spec_research.decoding as decoding
    original_feed = decoding.CachedModel.feed; original_crop = decoding.CachedModel.crop
    by_domain = {d: sorted([r for r in tt if r['domain'] == d], key=lambda r: (len(r['prompt_ids']), r['id'])) for d in DOMAINS}
    sample = [xs[int((len(xs)-1)*q)] for xs in by_domain.values() for q in quantiles]
    entries = []
    for r in sample:
        if method in ('DOMAIN', 'ROUTE-TASK', 'ROUTE-ACC'): draft.set_adapter(choose(method, r, fixed, clf))
        events = []; crops = []; round_state = {'round': 0}
        def feed(instance, ids):
            pre = instance.length() == 0
            who = 'target' if instance.model is target else 'draft'
            component = who+'_prefill' if pre else ('verify' if who == 'target' else 'draft')
            if who == 'target' and not pre: round_state['round'] += 1
            # Draft calls precede that round's target verification.
            round_id = 0 if component == 'target_prefill' else (round_state['round']+1 if who == 'draft' else round_state['round'])
            a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
            a.record(); t = time.perf_counter(); value = original_feed(instance, ids); wall = (time.perf_counter()-t)*1000; b.record()
            events.append((a, b, dict(component=component, round=round_id, input_tokens=len(ids), cpu_call_wall_ms=wall)))
            return value
        def crop(instance, n):
            t = time.perf_counter(); value = original_crop(instance, n)
            crops.append(dict(model='target' if instance.model is target else 'draft', round=round_state['round'], cpu_ms=(time.perf_counter()-t)*1000))
            return value
        decoding.CachedModel.feed = feed; decoding.CachedModel.crop = crop
        try:
            result = speculative(target, draft, r['prompt_ids'], 256, eos, L)
        finally:
            decoding.CachedModel.feed = original_feed; decoding.CachedModel.crop = original_crop
        assert result['tokens'] == previous[repeat, method, L, r['id']]['ar']['tokens'], 'Profiling changed full token output'
        sync()
        calls = [dict(**data, gpu_ms=a.elapsed_time(b)) for a, b, data in events]
        components = {k: sum(c['gpu_ms'] for c in calls if c['component'] == k) for k in ('target_prefill', 'draft_prefill', 'draft', 'verify')}
        entries.append(dict(id=r['id'], domain=r['domain'], prompt_tokens=len(r['prompt_ids']), output_tokens=len(result['tokens']),
                            decoder_profile_latency_ms=result['latency_ms'], components_gpu_ms=components, calls=calls, cache_cpu_ms=sum(x['cpu_ms'] for x in crops),
                            cache_calls=crops, rounds=result['rounds'], token_match=True))
    atomic(out/'component_profile.json', dict(state='complete', requests=entries, sample_count=len(sample), excluded_from_speed_metrics=True,
           policy='Separate deterministic replay with CUDA Events around feed, CPU wall around crop; no per-call synchronize added',
           limitations='Feed Events include GPU feed work and stream scheduling; cache timing is CPU only; logits selection, acceptance bookkeeping, gaps and profiling overhead remain outside component sums. No component sum is claimed equal to normal request latency.'))


def online_summary(scale):
    out = BASE/'online'/scale
    configs = [read(p) for p in (out/'configurations').glob('*/complete.json')]
    grouped = collections.defaultdict(list)
    for c in configs: grouped[c['method'], c['L']].append(c)
    table = []
    for (m, L), xs in sorted(grouped.items()):
        assert len({x['repeat'] for x in xs}) == len(xs)
        table.append(dict(method=m, L=L, completed_repeats=len(xs), state='complete' if len(xs) == 3 else 'partial',
                          median_macro_speedup_geo=statistics.median(x['macro_speedup_geo'] for x in xs),
                          repeats=sorted(xs, key=lambda x: x['repeat'])))
    chosen = {}
    for m in ONLINE:
        xs = [r for r in table if r['method'] == m and r['state'] == 'complete']
        if {r['L'] for r in xs} == {4, 8, 12}:
            win = max(xs, key=lambda r: (r['median_macro_speedup_geo'], -r['L']))
            chosen[m] = dict(L=win['L'], validation_median_macro_speedup_geo=win['median_macro_speedup_geo'])
    atomic(out/'summary.json', dict(state='complete' if len(configs) == 63 else 'partial', expected_configurations=63,
           completed_configurations=len(configs), table=table, selected_lengths=chosen, training_seed=42, timing_repeats=3,
           note='Local same-GPU sequential B1, real online draft/verify. Three timing repeats are not three training seeds. No WAN/SLO/async claims. Final tests remain locked.'))
    baseline_configs = [read(p) for p in (out/'baselines').glob('*/complete.json')]
    wall = [r['configuration_wall_seconds_with_profile'] for r in configs if 'configuration_wall_seconds_with_profile' in r]
    ar_wall = [r['configuration_wall_seconds'] for r in baseline_configs]
    if wall and ar_wall:
        atomic(out/'work_estimate.json', dict(at=now(), estimate_only=True, scale=scale,
               completed_SD_configurations=len(configs), completed_AR_configurations=len(baseline_configs),
               measured_SD_config_wall_seconds=wall, measured_AR_config_wall_seconds=ar_wall,
               estimated_remaining_online_hours=((63-len(configs))*statistics.median(wall)+(9-len(baseline_configs))*statistics.median(ar_wall))/3600,
               limitations='Extrapolates this scale using completed configurations. Unmeasured methods, L values, thermal/clock variation and profiling/resume overhead may differ. Excludes unfinished quality/router work and the other scale; not a full-run measurement.'))
    complete_lengths = {L for L in (4,8,12) if sum(x['L'] == L for x in configs) == 21}
    needed = [L for L in complete_lengths if not (out/f'paired_methods_L{L}.json').exists()]
    if needed:
        slim = [dict(id=x['id'], domain=x['domain'], repeat=x['repeat'], method=x['method'], L=x['L'], sd_ms=x['sd']['latency_ms'])
                for x in journal(out/'requests.jsonl', RECOVERY) if x['L'] in needed]
        for L in needed:
            comparisons = []
            for m in ('DOMAIN', 'FIXEDBEST', 'ROUTE-TASK', 'ROUTE-ACC'):
                for ref in ('mix16','mix48'):
                    repeats = [paired_latency_ratio(slim,m,ref,rep,L) for rep in range(3)]
                    comparisons.append(dict(candidate=m, reference=ref, median_macro_speedup_geo=statistics.median(x['macro_speedup_geo'] for x in repeats), repeats=repeats))
            atomic(out/f'paired_methods_L{L}.json', dict(state='complete', L=L, comparisons=comparisons,
                   note='Direct latency comparison: reference total / candidate total. 3 timing repeats retained separately; no pooling as independent training seeds.'))


def finalize_if_complete():
    if not all((BASE/'online'/s/'summary.json').exists() and read(BASE/'online'/s/'summary.json')['state'] == 'complete' for s in ('0.6B', '4B')):
        return
    validate_gate(); selections = {}; stages = {}; raw_counts = {}
    for s in ('0.6B', '4B'):
        for m in METHODS:
            sel = selection(s, m); complete = read(ROOT/'adapters'/s/m/'seed42'/RUN/'complete.json')
            n = 6000 if m.startswith('mix') else 2000
            assert len(complete['exposure_counts']) == n and set(complete['exposure_counts'].values()) == {2}
            assert complete['steps'] == (750 if n == 6000 else 250)
            selections[s+'.'+m] = dict(selected_epoch=sel['selected_epoch'], adapter_path=sel['adapter_path'],
                                      trained_steps=complete['steps'], exposure_count_per_sample=2, evidence=sel['evidence'],
                                      training={k:v for k,v in complete.items() if k != 'exposure_counts'})
        specialized_tokens = sum(selections[s+'.'+m]['training']['training_tokens'] for m in DOMAINS)
        assert specialized_tokens == selections[s+'.mix16']['training']['training_tokens'] == selections[s+'.mix48']['training']['training_tokens']
        quality_report(s)
        quality = read(BASE/'quality'/s/'quality_by_length.json')
        assert set(quality) == {'4', '8', '12'} and all(x['state'] == 'complete' and x['completed_matrix_requests'] == 600 for x in quality.values())
        routing = read(BASE/'routing'/s/'router_validation.json')
        assert routing['state'] == 'complete' and routing['train_requests']+routing['val_requests']+routing['excluded_no_public_window'] == 1500
        keys = set()
        for x in journal(BASE/'online'/s/'requests.jsonl', RECOVERY):
            assert x['token_match'] and x['eos_match'] and x['truncation_match'] and x['ar']['tokens'] == x['sd']['tokens']
            assert x['partition'] == 'adapter_val'; keys.add(online_key(x))
        expected = {(rep, m, L, r['id']) for rep in range(3) for m in ONLINE for L in (4,8,12) for r in records()}
        assert keys == expected
        profiles = list((BASE/'online'/s/'configurations').glob('*/component_profile.json'))
        assert len(profiles) == 63 and all(read(p)['sample_count'] == 12 for p in profiles)
        raw_counts[s] = dict(online_paired_comparisons=len(keys), independently_measured_AR_requests=5400,
                             configurations=63, component_profile_requests=63*12)
        stages[s] = dict(quality=quality, router_validation=routing, online=read(BASE/'online'/s/'summary.json'),
                         reused_deployment_report=str(ROOT/'results/overnight-20261017/deployment'/s/'report.json'))
    report = dict(state='complete', scope='Requested finite seed42 validation continuation only', completed_at=now(),
                  interpreter=sys.executable, program_sha256=digest(ROOT/'program.md'), protected_source_hashes=sources(),
                  selections=selections, stages=stages, counts=raw_counts,
                  correctness='All75600 SD comparisons exactly match measured target AR tokens/EOS/truncation. Oracle offline only.',
                  AR_measurement_policy='Each scale/L/repeat has a separate target-only AR8 process and600 measurements, shared across7 matched method comparisons. Reuse is explicit; no duplicated timings are counted as independent measurements.',
                  deployment_limitation='Historical BF16 merged-copy output differences preserved; do not claim mathematically identical draft behavior.',
                  unfinished_original_program=['seed43/44 replication', 'locked final ID and external tests', '3090 edge profiles', 'actual WAN/SLO/async second research point'],
                  results_directory=str(BASE), notes=['Checkpoint and L selection use validation; these are not held-out final-test results.',
                  'CUDA component profiling is separate from normal timing. Local online draft/verify does not simulate network transmission.'])
    atomic(BASE/'final_report.json', report)
    cfg = read(ROOT/'manifests/continuation_config.json')
    atomic(pathlib.Path(cfg['autoresearch_directory'])/'handoff.json', dict(state='finite_seed42_results_verified', report=str(BASE/'final_report.json'),
           next_action='Coordinator verifies final job and releases GPU; heartbeat confirms no residue and pauses', program_sha256=digest(ROOT/'program.md')))


def online(scale, lengths, lock_fd):
    out = BASE/'online'/scale
    tt = validate_teacher(records())
    fingerprint = bind(out, scale, tt, dict(acceptance_router_sha256=digest(router_path(scale))))
    for L in lengths:
        for repeat in range(3):
            ar_dir = out/'baselines'/f'L{L}.repeat{repeat}'
            if not (ar_dir/'complete.json').exists():
                cmd = [SPEC+'/bin/python', '-u', __file__, 'ar-config', '--scale', scale, '--lengths', str(L),
                       '--repeat', str(repeat), '--managed-lock-fd', str(lock_fd)]
                result = subprocess.run(cmd, cwd=ROOT, pass_fds=(lock_fd,))
                assert result.returncode == 0, f'Target-only AR8 failed: L{L}/repeat{repeat}'
            order = list(ONLINE); random.Random(42+repeat).shuffle(order)
            for method in order:
                cfgdir = out/'configurations'/f'L{L}.repeat{repeat}.{method}'
                if (cfgdir/'complete.json').exists() and (cfgdir/'component_profile.json').exists():
                    assert read(cfgdir/'complete.json')['requests'] == 600 and read(cfgdir/'complete.json')['fingerprint'] == fingerprint
                    assert read(cfgdir/'component_profile.json')['sample_count'] == 12
                    continue
                cmd = [SPEC+'/bin/python', '-u', __file__, 'online-config', '--scale', scale, '--method', method,
                       '--lengths', str(L), '--repeat', str(repeat), '--managed-lock-fd', str(lock_fd)]
                print(json.dumps(dict(event='new_process_configuration', method=method, L=L, repeat=repeat, command=cmd)), flush=True)
                # Same process group as this job: coordinator/watchdog can clean up all children.
                result = subprocess.run(cmd, cwd=ROOT, pass_fds=(lock_fd,))
                assert result.returncode == 0, f'Online configuration failed: {method}/L{L}/repeat{repeat}'
                online_summary(scale)
    online_summary(scale)
    finalize_if_complete()


def preliminary_online(scale, lock_fd):
    out = online_out(scale, True); tt = validate_teacher(preliminary_records())
    fp = bind(out, scale, tt, dict(acceptance_router_sha256=digest(router_path(scale))))
    subset = dict(scope='User-authorized preliminary24, not the full formal validation protocol', seed=20261017,
                  partition='adapter_val', requests=24, per_domain=8, L=8, timing_repeats=1,
                  selection='Uniform sampling within source domains, using prompt records only; same ordered IDs for all methods and both scales',
                  ids=[r['id'] for r in tt], prompt_hashes={r['id']:r['prompt_hash'] for r in tt})
    p = out/'subset_manifest.json'
    if p.exists(): assert read(p) == subset
    else: atomic(p, subset)
    ar_dir = out/'baselines'/'L8.repeat0'
    if not (ar_dir/'complete.json').exists():
        command = [SPEC+'/bin/python','-u',__file__,'ar-config','--scale',scale,'--lengths','8','--repeat','0',
                   '--preliminary','--managed-lock-fd',str(lock_fd)]
        child = subprocess.run(command,cwd=ROOT,pass_fds=(lock_fd,)); assert child.returncode == 0
    assert read(ar_dir/'complete.json')['requests'] == 24 and read(ar_dir/'complete.json')['fingerprint'] == fp
    order = list(ONLINE); random.Random(42).shuffle(order)
    for method in order:
        cfgdir = out/'configurations'/f'L8.repeat0.{method}'
        if (cfgdir/'complete.json').exists() and (cfgdir/'component_profile.json').exists():
            assert read(cfgdir/'complete.json')['requests'] == 24 and read(cfgdir/'complete.json')['fingerprint'] == fp
            assert read(cfgdir/'component_profile.json')['sample_count'] == 3
            continue
        command = [SPEC+'/bin/python','-u',__file__,'online-config','--scale',scale,'--method',method,'--lengths','8',
                   '--repeat','0','--preliminary','--managed-lock-fd',str(lock_fd)]
        child = subprocess.run(command,cwd=ROOT,pass_fds=(lock_fd,)); assert child.returncode == 0, f'Preliminary24 config failed: {method}'
    index = unique_index(journal(out/'requests.jsonl',RECOVERY), online_key)
    expected = {(0,m,8,r['id']) for m in ONLINE for r in tt}; assert set(index) == expected
    assert all(x['fingerprint'] == fp and x['token_match'] and x['ar']['tokens'] == x['sd']['tokens'] and x['eos_match'] and x['truncation_match'] for x in index.values())
    configurations = [read(out/'configurations'/f'L8.repeat0.{m}'/'complete.json') for m in ONLINE]
    slim = [dict(id=x['id'],domain=x['domain'],repeat=0,L=8,method=x['method'],sd_ms=x['sd']['latency_ms']) for x in index.values()]
    comparisons = [paired_latency_ratio(slim,m,ref,0,8,domain_n=8) for m in ('DOMAIN','FIXEDBEST','ROUTE-TASK','ROUTE-ACC') for ref in ('mix16','mix48')]
    atomic(out/'summary.json',dict(state='complete',scope='preliminary24 only; no formal600-request completion claim',scale=scale,
           fingerprint=fp,requests_per_configuration=24,requests_per_domain=8,SD_comparisons=168,independently_measured_AR_requests=24,
           L=8,timing_repeats=1,configurations=configurations,direct_method_comparisons=comparisons,
           limitations=['Small selected validation subset; eight requests per domain cannot support a strong general improvement claim.',
           'One timing repeat: request-bootstrap intervals do not quantify run-to-run timing variability.',
           'No final tests or seed43/44, no network/SLO simulation, and no online Oracle.',
           'Three separate component-profile requests per configuration, excluded from normal timings.'],program_sha256=digest(ROOT/'program.md')))
    progress(out,job='preliminary24',state='complete',SD_comparisons=168,independent_AR_requests=24)


if __name__ == '__main__':
    assert_spec(); torch.set_num_threads(4); seed_all(42)
    p = argparse.ArgumentParser(); p.add_argument('job', choices=['quality', 'router', 'online', 'online-config', 'ar-config','preliminary'])
    p.add_argument('--scale', choices=['0.6B', '4B'], required=True); p.add_argument('--lengths', default='8')
    p.add_argument('--managed-lock-fd', type=int, required=True); p.add_argument('--method', choices=ONLINE); p.add_argument('--repeat', type=int)
    p.add_argument('--preliminary',action='store_true')
    a = p.parse_args(); lengths = tuple(int(x) for x in a.lengths.split(',')); assert all(x in (4, 8, 12) for x in lengths)
    validate_gate()
    with gpu_lock(a.managed_lock_fd):
        if a.job == 'quality': quality(a.scale, lengths, a.managed_lock_fd)
        elif a.job == 'router': acceptance_router(a.scale)
        elif a.job == 'online': online(a.scale, lengths, a.managed_lock_fd)
        elif a.job == 'preliminary': preliminary_online(a.scale,a.managed_lock_fd)
        elif a.job == 'ar-config':
            assert a.repeat in range(3) and len(lengths) == 1
            ar_config(a.scale, lengths[0], a.repeat,a.preliminary)
        else:
            assert a.method in ONLINE and a.repeat in range(3) and len(lengths) == 1
            online_config(a.scale, a.method, lengths[0], a.repeat, a.managed_lock_fd,a.preliminary)

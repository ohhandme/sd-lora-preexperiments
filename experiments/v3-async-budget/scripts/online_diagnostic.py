#!/root/miniconda3/envs/spec/bin/python
"""Small independent GPU online diagnostic after the frozen negative P2 gate."""

import argparse
import hashlib
import json
import os
import pathlib
import sys
import time

ROOT = pathlib.Path('/root/autodl-tmp/sd-lora-preexp')
V2 = ROOT / 'experiments/v2-routing'
V3 = ROOT / 'experiments/v3-async-budget'
OUT = V3 / 'results/online_diagnostic'
sys.path[:0] = [str(ROOT), str(V2 / 'scripts')]

import torch
from peft import PeftModel
from spec_research.common import assert_spec, atomic, digest, load_model, now, objhash
from spec_research.decoding import CachedModel, ar, compare, margin, speculative, sync, top
from train import DOMAINS, records
from v2_online import selected, eos


def subset():
    source = records('mix16', 'router_train')
    rows = []
    for domain in DOMAINS:
        candidates = [r for r in source if r['domain'] == domain]
        candidates.sort(key=lambda r: hashlib.sha256(
            ('v3-online-diagnostic-20261018|' + r['id']).encode()).hexdigest())
        rows.extend(candidates[:3])
    assert len(rows) == 15 and len({r['id'] for r in rows}) == 15
    binding = {'seed': 'v3-online-diagnostic-20261018',
               'ids': [r['id'] for r in rows],
               'teacher_shas': [r['record_sha'] for r in rows],
               'pilot_sha256': digest(V2 / 'data/processed/pilot_manifest.jsonl'),
               'policy': 'BF16 B1 max_new128; AR, fixed L8, previous-acceptance L4/8; one timing repeat'}
    path = OUT / 'subset_manifest.json'
    if path.exists():
        assert json.loads(path.read_text()) == binding
    else:
        atomic(path, binding)
    return rows, objhash(binding)


@torch.inference_mode()
def history_decode(target, draft, prompt, max_new, eos_ids):
    sync()
    start = time.perf_counter()
    tc, dc = CachedModel(target), CachedModel(draft)
    committed, output, rounds, traces = list(prompt), [], [], []
    first = None
    first_logits = tc.feed(prompt)[-1]
    prefill_calls = tc.calls
    previous_accepted = None
    while len(output) < max_new:
        base = len(committed)
        L = 8 if previous_accepted is None or previous_accepted > 2 else 4
        proposal, draft_prefix = [], list(committed)
        for _ in range(min(L, max_new - len(output))):
            token = top(dc.next(draft_prefix))
            proposal.append(token)
            draft_prefix.append(token)
            if token in eos_ids:
                break
        if first_logits is not None:
            assert tc.length() == base
            logits = torch.cat((first_logits.unsqueeze(0), tc.feed(proposal)), dim=0)
            first_logits = None
        else:
            assert tc.length() == base - 1
            logits = tc.feed([committed[-1]] + proposal)
        accepted, emitted, done = 0, [], False
        for j, token in enumerate(proposal):
            predicted = top(logits[j])
            traces.append(margin(logits[j]))
            if token == predicted:
                accepted += 1
                emitted.append(token)
            else:
                emitted.append(predicted)
            if emitted[-1] in eos_ids or len(output) + len(emitted) >= max_new:
                done = True
                break
            if token != predicted:
                break
        else:
            if len(output) + len(emitted) < max_new:
                bonus = top(logits[len(proposal)])
                emitted.append(bonus)
                traces.append(margin(logits[len(proposal)]))
                if bonus in eos_ids:
                    done = True
        assert emitted
        output.extend(emitted)
        committed.extend(emitted)
        tc.crop(len(committed) - 1)
        dc.crop(min(dc.length(), base + accepted, len(committed) - 1))
        rounds.append({'Lr': len(proposal), 'ar': accepted, 'nr': len(emitted),
                       'prefix_len': base, 'target_cache': tc.length(),
                       'draft_cache': dc.length()})
        previous_accepted = accepted
        if first is None:
            first = time.perf_counter()
        if done:
            break
    sync()
    end = time.perf_counter()
    return {'tokens': output, 'latency_ms': (end - start) * 1000,
            'ttft_ms': ((first or end) - start) * 1000,
            'decode_seconds': end - (first or end), 'rounds': rounds,
            'target_prefill_calls': prefill_calls,
            'target_verify_calls': tc.calls - prefill_calls,
            'target_calls': tc.calls, 'draft_calls': dc.calls,
            'margins': traces}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', required=True, choices=('AR', 'L8', 'H4-8'))
    args = parser.parse_args()
    assert_spec()
    torch.set_num_threads(4)
    rows, subset_sha = subset()
    target = load_model('8B').eval()
    draft = None
    if args.method != 'AR':
        draft = PeftModel.from_pretrained(load_model('0.6B'), selected('mix80'),
                                           adapter_name='mix80', is_trainable=False).eval()
    eos_ids = eos(target)
    warm_rows = records('code', 'expert_train')[:5]
    for row in warm_rows:
        if args.method == 'AR':
            ar(target, row['prompt_ids'], 16, eos_ids)
        elif args.method == 'L8':
            speculative(target, draft, row['prompt_ids'], 16, eos_ids, 8)
        else:
            history_decode(target, draft, row['prompt_ids'], 16, eos_ids)
    for index, row in enumerate(rows):
        key = hashlib.sha256(row['id'].encode()).hexdigest() + '.json'
        path = OUT / args.method / key
        if path.exists():
            saved = json.loads(path.read_text())
            assert saved['id'] == row['id'] and saved['subset_sha'] == subset_sha
            assert saved['token_eos_truncation_equal']
            continue
        baseline = None
        if args.method != 'AR':
            baseline = json.loads((OUT / 'AR' / key).read_text())
            assert baseline['id'] == row['id'] and baseline['subset_sha'] == subset_sha
        torch.cuda.reset_peak_memory_stats()
        if args.method == 'AR':
            result = ar(target, row['prompt_ids'], 128, eos_ids)
        elif args.method == 'L8':
            result = speculative(target, draft, row['prompt_ids'], 128, eos_ids, 8)
        else:
            result = history_decode(target, draft, row['prompt_ids'], 128, eos_ids)
        mismatch = compare(baseline['result'], result) if baseline else None
        if mismatch is not None:
            atomic(OUT / 'failures' / args.method / key,
                   {'at': now(), 'id': row['id'], 'mismatch': mismatch,
                    'baseline': baseline['result'], 'candidate': result})
            raise RuntimeError(f'Complete token mismatch for {row["id"]}')
        tokens = result['tokens']
        assert tokens
        item = {'id': row['id'], 'domain': row['domain'], 'method': args.method,
                'subset_sha': subset_sha, 'result': result,
                'eos': tokens[-1] in eos_ids,
                'truncated': len(tokens) == 128 and tokens[-1] not in eos_ids,
                'peak_allocated': torch.cuda.max_memory_allocated(),
                'peak_reserved': torch.cuda.max_memory_reserved(),
                'token_eos_truncation_equal': True}
        if baseline:
            assert item['eos'] == baseline['eos'] and item['truncated'] == baseline['truncated']
        atomic(path, item)
        atomic(OUT / f'status.{args.method}.json',
               {'at': now(), 'state': 'running', 'pid': os.getpid(),
                'completed': index + 1, 'requested': len(rows)})
        print(json.dumps({'method': args.method, 'completed': index + 1}), flush=True)
    atomic(OUT / f'status.{args.method}.json',
           {'at': now(), 'state': 'complete', 'completed': len(rows),
            'requested': len(rows)})


if __name__ == '__main__':
    main()

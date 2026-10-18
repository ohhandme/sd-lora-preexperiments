#!/root/miniconda3/envs/spec/bin/python
"""V2 pilot feedback baselines with exact target verification.

META-UCB is a simple verification-reward bandit inspired by MetaSD, not a
reimplementation of that paper. COST-UCB adds a train-only task prior and a
measured switch/rebuild cost screen. Neither sees future target tokens.
"""

import argparse
import hashlib
import json
import math
import os
import pathlib
import pickle
import statistics
import sys
import time

ROOT = pathlib.Path('/root/autodl-tmp/sd-lora-preexp')
V2 = ROOT / 'experiments/v2-routing'
sys.path[:0] = [str(ROOT), str(V2 / 'scripts')]

import torch
from peft import PeftModel
from spec_research.common import assert_spec, atomic, digest, load_model, now
from spec_research.decoding import CachedModel, ar, compare, margin, sync, top
from train import DOMAINS, records
from online import OUT, baseline_path, eos, output_path, selected, subset

METHODS = ('META-UCB', 'COST-UCB')


def prior_from_train():
    result = {d: {e: [] for e in DOMAINS} for d in DOMAINS}
    for row in records('mix16', 'router_train'):
        key = hashlib.sha256(row['id'].encode()).hexdigest() + '.json'
        for expert in DOMAINS:
            item = json.loads((V2 / 'results/routing/diagnostics' / expert / key).read_text())
            score = item['diagnostic']['mean_a']
            if score is not None:
                result[row['domain']][expert].append(score)
    assert all(len(result[d][e]) >= 90 for d in DOMAINS for e in DOMAINS)
    return {d: {e: statistics.mean(result[d][e]) for e in DOMAINS} for d in DOMAINS}


class Policy:
    def __init__(self, method, initial, prior, cost):
        assert method in METHODS and initial in DOMAINS
        self.method = method
        self.current = initial
        self.initial = initial
        self.prior = prior
        self.cost = cost
        self.n = {e: 0 for e in DOMAINS}
        self.reward = {e: 0.0 for e in DOMAINS}
        self.switches = 0
        self.first_draft_prefill_ms = None

    def choose(self, turn, context_length, prompt_length):
        if turn == 0:
            return self.current
        if self.method == 'META-UCB':
            if self.switches >= 8:
                return self.current
            unexplored = [e for e in DOMAINS if self.n[e] == 0]
            if unexplored:
                chosen = unexplored[0]
            else:
                chosen = max(DOMAINS, key=lambda e: self.reward[e] / self.n[e] +
                             1.5 * math.sqrt(math.log(turn + 1) / self.n[e]))
        else:
            chosen = self.current
            if self.switches < 2:
                def expected(expert):
                    return (4.0 * self.prior[expert] + self.reward[expert]) / (4.0 + self.n[expert])
                current_score = expected(self.current)
                candidate = max((e for e in DOMAINS if e != self.current),
                                key=lambda e: expected(e) + 0.8 / math.sqrt(1 + self.n[e]))
                optimistic_gain_a = expected(candidate) + 0.8 / math.sqrt(1 + self.n[candidate]) - current_score
                mix_a = self.cost['offline_mix80_mean_a']
                saving_ms = max(0.0, optimistic_gain_a) * self.cost['target_verify_8_ms_median'] / (1 + mix_a)
                # Switching invalidates the draft KV. A fresh draft prefill is
                # mandatory and its measured warm-start cost enters the screen.
                rebuild_ms = (self.first_draft_prefill_ms or self.cost['target_verify_8_ms_median']) * \
                    max(1.0, context_length / max(1, prompt_length))
                switch_ms = self.cost['adapter_switch_ms_median'] + rebuild_ms
                if saving_ms > switch_ms:
                    chosen = candidate
        if chosen != self.current:
            self.current = chosen
            self.switches += 1
        return chosen

    def observe(self, expert, accepted):
        assert expert in DOMAINS and 0 <= accepted <= 8
        self.n[expert] += 1
        self.reward[expert] += accepted


@torch.inference_mode()
def feedback_speculative(target, draft, prompt, max_new, eos_ids, L, policy):
    sync()
    start = time.perf_counter()
    tc = CachedModel(target)
    dc = None
    active = None
    committed = list(prompt)
    out = []
    rounds = []
    traces = []
    first = None
    draft_calls = 0
    first_logits = tc.feed(prompt)[-1]
    prefill_calls = tc.calls
    while len(out) < max_new:
        base = len(committed)
        expert = policy.choose(len(rounds), base, len(prompt))
        if expert != active:
            draft.set_adapter(expert)
            dc = CachedModel(draft)  # Never reuse a KV made by another adapter.
            active = expert
        proposal = []
        draft_prefix = list(committed)
        for j in range(min(L, max_new - len(out))):
            before = time.perf_counter() if policy.first_draft_prefill_ms is None else None
            token = top(dc.next(draft_prefix))
            draft_calls += 1
            if before is not None and j == 0:
                sync()
                policy.first_draft_prefill_ms = (time.perf_counter() - before) * 1000
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
        accepted = 0
        emitted = []
        done = False
        for j, token in enumerate(proposal):
            predicted = top(logits[j])
            traces.append(margin(logits[j]))
            if token == predicted:
                accepted += 1
                emitted.append(token)
            else:
                emitted.append(predicted)
            if emitted[-1] in eos_ids or len(out) + len(emitted) >= max_new:
                done = True
                break
            if token != predicted:
                break
        else:
            if len(out) + len(emitted) < max_new:
                token = top(logits[len(proposal)])
                emitted.append(token)
                traces.append(margin(logits[len(proposal)]))
                if token in eos_ids:
                    done = True
        assert emitted
        out.extend(emitted)
        committed.extend(emitted)
        tc.crop(len(committed) - 1)
        dc.crop(min(dc.length(), base + accepted, len(committed) - 1))
        policy.observe(expert, accepted)
        rounds.append({'Lr': len(proposal), 'ar': accepted, 'nr': len(emitted),
                       'expert': expert, 'prefix_len': base,
                       'target_cache': tc.length(), 'draft_cache': dc.length()})
        if first is None:
            first = time.perf_counter()
        if done:
            break
    sync()
    end = time.perf_counter()
    return {'tokens': out, 'latency_ms': (end - start) * 1000,
            'ttft_ms': ((first or end) - start) * 1000,
            'decode_seconds': end - (first or end), 'rounds': rounds,
            'target_prefill_calls': prefill_calls,
            'target_verify_calls': tc.calls - prefill_calls,
            'target_calls': tc.calls, 'draft_calls': draft_calls,
            'margins': traces, 'switches': policy.switches,
            'first_draft_prefill_ms': policy.first_draft_prefill_ms}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=METHODS, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    assert_spec()
    torch.set_num_threads(4)
    rows, subset_sha = subset()
    prompts = {r['record_id']: r['prompt'] for r in
               (json.loads(line) for line in (V2 / 'data/processed/pilot_manifest.jsonl').read_text().splitlines() if line)}
    with (V2 / 'results/routing/task_router.pkl').open('rb') as stream:
        task_router = pickle.load(stream)
    prior = prior_from_train()
    cost = json.loads((V2 / 'results/cost_gate/summary.json').read_text())
    target = load_model('8B').eval()
    draft = load_model('0.6B')
    for index, expert in enumerate(DOMAINS):
        if index == 0:
            draft = PeftModel.from_pretrained(draft, selected(expert),
                                               adapter_name=expert, is_trainable=False)
        else:
            draft.load_adapter(selected(expert), adapter_name=expert, is_trainable=False)
    draft.eval()
    eos_ids = eos(target)
    method = args.method
    # Disjoint expert_train warmups, not scored or saved as pilot requests.
    for row in records('code', 'expert_train')[:3]:
        feedback_speculative(target, draft, row['prompt_ids'], 16, eos_ids, 8,
                             Policy(method, 'code', prior['code'], cost))
    if args.smoke:
        smoke_rows = [next(r for r in records(d, 'expert_train')
                           if len(r['response_ids']) >= 80) for d in DOMAINS[:3]]
        assert len(smoke_rows) == 3
        outcomes = []
        for row in smoke_rows:
            reference = ar(target, row['prompt_ids'], 64, eos_ids)
            candidate = feedback_speculative(target, draft, row['prompt_ids'], 64,
                                             eos_ids, 8,
                                             Policy(method, row['domain'],
                                                    prior[row['domain']], cost))
            mismatch = compare(reference, candidate)
            if mismatch is not None:
                atomic(V2 / 'results/smoke' / f'feedback_{method}_failure.json',
                       {'id': row['id'], 'mismatch': mismatch})
                raise RuntimeError('Feedback smoke token mismatch')
            outcomes.append({'id': row['id'], 'tokens': len(candidate['tokens']),
                             'switches': candidate['switches']})
        atomic(V2 / 'results/smoke' / f'feedback_{method}.json',
               {'state': 'complete', 'method': method, 'at': now(),
                'outcomes': outcomes})
        print(json.dumps({'state': 'complete', 'method': method,
                          'outcomes': outcomes}), flush=True)
        return
    completed = 0
    for row in rows:
        path = output_path(method, row)
        if path.exists():
            item = json.loads(path.read_text())
            assert item['id'] == row['id'] and item['subset_sha'] == subset_sha
            assert item['token_eos_truncation_equal']
            completed += 1
            continue
        baseline = json.loads(baseline_path(row).read_text())
        assert baseline['id'] == row['id'] and baseline['subset_sha'] == subset_sha
        start = time.perf_counter()
        task_choice = str(task_router.predict([prompts[row['id']]])[0])
        route_ms = (time.perf_counter() - start) * 1000
        assert task_choice in DOMAINS
        torch.cuda.reset_peak_memory_stats()
        result = feedback_speculative(target, draft, row['prompt_ids'], 256, eos_ids, 8,
                                      Policy(method, task_choice, prior[task_choice], cost))
        mismatch = compare(baseline['result'], result)
        if mismatch is not None:
            atomic(OUT / 'failures' / method /
                   (hashlib.sha256(row['id'].encode()).hexdigest() + '.json'),
                   {'state': 'failed', 'at': now(), 'id': row['id'],
                    'method': method, 'mismatch': mismatch,
                    'baseline': baseline['result'], 'candidate': result})
            raise RuntimeError('Full token mismatch; no request committed')
        tokens = result['tokens']
        assert tokens
        item = {'id': row['id'], 'domain': row['domain'], 'method': method,
                'subset_sha': subset_sha, 'choice': task_choice,
                'routing_ms': route_ms, 'result': result,
                'eos': tokens[-1] in eos_ids,
                'truncated': len(tokens) == 256 and tokens[-1] not in eos_ids,
                'peak_allocated': torch.cuda.max_memory_allocated(),
                'peak_reserved': torch.cuda.max_memory_reserved(),
                'token_eos_truncation_equal':
                    tokens == baseline['result']['tokens'] and
                    (tokens[-1] in eos_ids) == baseline['eos'] and
                    (len(tokens) == 256 and tokens[-1] not in eos_ids) == baseline['truncated']}
        assert item['token_eos_truncation_equal']
        atomic(path, item)
        completed += 1
        atomic(V2 / 'status' / f'online.{method}.json',
               {'state': 'running', 'at': now(), 'pid': os.getpid(),
                'method': method, 'completed': completed, 'requested': 40})
        print(json.dumps({'method': method, 'completed': completed, 'requested': 40}), flush=True)
    atomic(V2 / 'status' / f'online.{method}.json',
           {'state': 'complete', 'at': now(), 'method': method,
            'completed': completed, 'requested': 40})


if __name__ == '__main__':
    main()

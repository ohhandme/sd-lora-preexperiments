#!/root/miniconda3/envs/spec/bin/python
"""Isolated, resumable v2 pilot teacher; never alters the v1 teacher store."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import signal
import sys
import time

ROOT = pathlib.Path('/root/autodl-tmp/sd-lora-preexp')
V2 = ROOT / 'experiments/v2-routing'
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts/overnight'), str(ROOT / 'scripts/diagnostics')]

import torch
from spec_research.common import assert_spec, atomic, append, assets, digest, load_model, now, objhash, prompt_ids, tokenizer
from overnight_engine import optimize_batch_norm
from teacher_benchmark import isolated_batch, lean_single

MANIFEST = V2 / 'data/processed/pilot_manifest.jsonl'
FROZEN = V2 / 'data/processed/manifest.json'
OUTPUT = V2 / 'data/teacher/pilot'
MAX_NEW = 512
stop = False
STATUS = V2 / 'status/teacher.json'
GATE = V2 / 'status/teacher_gate.json'
LOG = V2 / 'logs/teacher_batches.jsonl'


def request_stop(*_):
    global stop
    stop = True


def entries():
    frozen = json.loads(FROZEN.read_text())
    assert frozen['state'] == 'frozen' and frozen['pilot_count'] == 2120
    assert digest(MANIFEST) == frozen['pilot_sha256']
    rows = [json.loads(line) for line in MANIFEST.read_text().splitlines() if line]
    assert len(rows) == 2120 and len({r['record_id'] for r in rows}) == 2120
    allowed = {'expert_train', 'adapter_val'}
    return [r for r in rows if r['partition'] in allowed], frozen


def path_for(r):
    return OUTPUT / r['domain'] / r['partition'] / (hashlib.sha256(r['record_id'].encode()).hexdigest() + '.json')


def validate_done(r, path, manifest_sha, prompt):
    if not path.exists():
        return False
    item = json.loads(path.read_text())
    assert item['record_id'] == r['record_id']
    assert item['source_revision'] == r['source_revision']
    assert item['prompt_sha256'] == r['prompt_sha256']
    assert item['prompt_ids'] == prompt
    assert item['manifest_sha256'] == manifest_sha
    assert 0 < len(item['response_ids']) <= MAX_NEW
    assert item['record_sha256'] == objhash({k: v for k, v in item.items() if k != 'record_sha256'})
    return True


def state(value, **details):
    atomic(STATUS, {'state': value, 'at': now(), 'pid': os.getpid(), **details})


def main():
    global STATUS, GATE, LOG
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--gate-only', action='store_true')
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--shard-count', type=int, default=1)
    args = parser.parse_args()
    assert args.batch in (1, 4, 8, 16)
    assert args.shard_count in (1, 2) and 0 <= args.shard_index < args.shard_count
    if args.shard_count > 1:
        STATUS = V2 / f'status/teacher.{args.shard_index}.json'
        GATE = V2 / f'status/teacher_gate.{args.shard_index}.json'
        LOG = V2 / f'logs/teacher_batches.{args.shard_index}.jsonl'
    assert_spec()
    torch.set_num_threads(4)
    signal.signal(signal.SIGUSR1, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    rows, frozen = entries()
    if args.shard_count > 1:
        rows = [r for r in rows if int(hashlib.sha256(r['record_id'].encode()).hexdigest()[:16], 16)
                % args.shard_count == args.shard_index]
    manifest_sha = frozen['pilot_sha256']
    tok = tokenizer()
    for r in rows:
        r['prompt_ids'] = prompt_ids(tok, r['prompt'])
        assert 0 < len(r['prompt_ids']) <= 1024
    eos_value = None
    state('loading', requested=len(rows), completed=0, batch=args.batch, manifest_sha256=manifest_sha)
    model = optimize_batch_norm(load_model('8B'), use_jit=False)
    eos_value = model.generation_config.eos_token_id
    eos = set([eos_value] if isinstance(eos_value, int) else eos_value)
    assert eos
    gate_rows = [next(r for r in rows if r['domain'] == d and r['partition'] == 'expert_train')
                 for d in ('code', 'translation', 'summary', 'qa', 'math')]
    # Add variable-length and duplicate-domain prompts so the selected batch
    # path is checked against actual full-length serial generation.
    gate_rows.extend([next(r for r in rows if r['domain'] == d and r['partition'] == 'adapter_val')
                      for d in ('code', 'summary', 'math')])
    gate_rows = gate_rows[:args.batch]
    start = time.perf_counter()
    baseline = [lean_single(model, r['prompt_ids'], MAX_NEW, eos) for r in gate_rows]
    actual = isolated_batch(model, [r['prompt_ids'] for r in gate_rows], MAX_NEW, eos)
    match = actual == baseline
    atomic(GATE, {'state': 'passed' if match else 'failed', 'at': now(), 'batch': args.batch,
                  'ids': [r['record_id'] for r in gate_rows],
                  'lengths': [len(x) for x in baseline],
                  'prompt_lengths': [len(r['prompt_ids']) for r in gate_rows],
                  'full_token_eos_truncation_equal': match, 'seconds': time.perf_counter() - start,
                  'manifest_sha256': manifest_sha, 'script_sha256': digest(__file__)})
    if not match:
        state('failed', reason='cross-task full-length batch mismatch')
        raise RuntimeError('Cross-task full-length batch/serial token mismatch')
    if args.gate_only:
        state('gate_passed', requested=len(rows), completed=0, batch=args.batch)
        return
    order = {name: index for index, name in enumerate(('code', 'translation', 'summary', 'qa', 'math'))}
    rows.sort(key=lambda r: (0 if r['partition'] == 'expert_train' else 1,
                             order[r['domain']], len(r['prompt_ids']), r['record_id']))
    done = {r['record_id'] for r in rows if validate_done(r, path_for(r), manifest_sha, r['prompt_ids'])}
    pending = [r for r in rows if r['record_id'] not in done]
    state('running', requested=len(rows), completed=len(done), batch=args.batch,
          manifest_sha256=manifest_sha, gate_sha256=digest(GATE))
    started = time.perf_counter()
    for offset in range(0, len(pending), args.batch):
        if stop:
            state('paused', requested=len(rows), completed=len(done), batch=args.batch)
            return
        group = pending[offset:offset + args.batch]
        t0 = time.perf_counter()
        outputs = isolated_batch(model, [r['prompt_ids'] for r in group], MAX_NEW, eos)
        audit_index = (offset // args.batch) % len(group)
        reference = lean_single(model, group[audit_index]['prompt_ids'], MAX_NEW, eos)
        if outputs[audit_index] != reference:
            failure = {'state': 'failed', 'at': now(), 'record_id': group[audit_index]['record_id'],
                       'batch': args.batch, 'actual': outputs[audit_index], 'reference': reference}
            atomic(V2 / 'status' / f'teacher_audit_failure.{args.shard_index}.json', failure)
            state('failed', requested=len(rows), completed=len(done), reason='full-length audit mismatch')
            raise RuntimeError('Production batch audit mismatch; batch not committed')
        for r, tokens in zip(group, outputs):
            assert tokens and len(tokens) <= MAX_NEW
            item = {'record_id': r['record_id'], 'domain': r['domain'], 'partition': r['partition'],
                    'source_revision': r['source_revision'], 'prompt_sha256': r['prompt_sha256'],
                    'prompt_ids': r['prompt_ids'], 'response_ids': tokens,
                    'eos': tokens[-1] in eos, 'truncated': len(tokens) == MAX_NEW and tokens[-1] not in eos,
                    'eos_ids': sorted(eos), 'manifest_sha256': manifest_sha,
                    'model_revision': assets()['models']['Qwen/Qwen3-8B']['revision'],
                    'runtime': {'engine': 'isolated_batch', 'batch': args.batch,
                                'audit': 'rotating-one-per-batch-full-output',
                                'script_sha256': digest(__file__)}}
            item['record_sha256'] = objhash(item)
            path = path_for(r)
            assert not path.exists(), f'Unexpected competing writer: {path}'
            atomic(path, item)
            done.add(r['record_id'])
        append(LOG, {'at': now(), 'ids': [r['record_id'] for r in group],
                     'output_tokens': sum(map(len, outputs)), 'seconds': time.perf_counter() - t0,
                     'audited_id': group[audit_index]['record_id'], 'audit_exact': True})
        state('running', requested=len(rows), completed=len(done), batch=args.batch,
              seconds=time.perf_counter() - started, manifest_sha256=manifest_sha)
        print(json.dumps({'completed': len(done), 'requested': len(rows), 'at': now()}), flush=True)
    assert len(done) == len(rows)
    state('complete', requested=len(rows), completed=len(done), batch=args.batch,
          seconds=time.perf_counter() - started, manifest_sha256=manifest_sha)


if __name__ == '__main__':
    main()

#!/root/miniconda3/envs/spec/bin/python
"""Five-task v2 pilot LoRA training and held-out checkpoint selection."""

from __future__ import annotations

import argparse
import collections
import gc
import hashlib
import json
import math
import os
import pathlib
import random
import signal
import statistics
import sys
import time

ROOT = pathlib.Path('/root/autodl-tmp/sd-lora-preexp')
V2 = ROOT / 'experiments/v2-routing'
sys.path.insert(0, str(ROOT))

import torch
from peft import PeftModel, get_peft_model_state_dict, set_peft_model_state_dict
from spec_research.common import (assert_spec, atomic, append, digest, load_model,
                                  now, objhash, seed_all)
from spec_research.decoding import public_prefix
from spec_research.training import attach, batch_record, geometry

DOMAINS = ('code', 'translation', 'summary', 'qa', 'math')
METHODS = (*DOMAINS, 'mix16', 'mix80')
PILOT = V2 / 'data/processed/pilot_manifest.jsonl'
MANIFEST = V2 / 'data/processed/manifest.json'
TEACHER = V2 / 'data/teacher/pilot'
stop = False


def request_stop(*_):
    global stop
    stop = True


def teacher_path(r):
    name = hashlib.sha256(r['record_id'].encode()).hexdigest() + '.json'
    return TEACHER / r['domain'] / r['partition'] / name


def records(method, partition):
    frozen = json.loads(MANIFEST.read_text())
    assert frozen['state'] == 'frozen' and digest(PILOT) == frozen['pilot_sha256']
    wanted = set(DOMAINS) if method.startswith('mix') else {method}
    rr = [json.loads(line) for line in PILOT.read_text().splitlines() if line]
    out = []
    for r in rr:
        if r['domain'] not in wanted or r['partition'] != partition:
            continue
        path = teacher_path(r)
        if not path.is_file():
            raise FileNotFoundError(path)
        t = json.loads(path.read_text())
        assert t['record_id'] == r['record_id'] and t['manifest_sha256'] == frozen['pilot_sha256']
        assert t['prompt_sha256'] == r['prompt_sha256']
        assert t['record_sha256'] == objhash({k: v for k, v in t.items() if k != 'record_sha256'})
        assert 0 < len(t['response_ids']) <= 512
        assert t['eos'] == (t['response_ids'][-1] in t['eos_ids'])
        assert t['truncated'] == (len(t['response_ids']) == 512 and not t['eos'])
        out.append({'id': r['record_id'], 'domain': r['domain'], 'partition': partition,
                    'prompt_ids': t['prompt_ids'], 'response_ids': t['response_ids'],
                    'record_sha': t['record_sha256']})
    expected = {'expert_train': 256, 'adapter_val': 48,
                'router_train': 96, 'router_val': 24}[partition] * len(wanted)
    assert len(out) == expected, (len(out), expected)
    assert len({x['id'] for x in out}) == len(out)
    return out


def order(rows, seed, epoch, mixed):
    rng = random.Random(seed + epoch)
    if not mixed:
        selected = rows.copy()
        rng.shuffle(selected)
        return selected
    groups = {d: [r for r in rows if r['domain'] == d] for d in DOMAINS}
    assert all(len(group) == len(groups[DOMAINS[0]]) for group in groups.values())
    for group in groups.values():
        rng.shuffle(group)
    out = []
    for index in range(len(groups[DOMAINS[0]])):
        names = list(DOMAINS)
        rng.shuffle(names)
        out.extend(groups[name][index] for name in names)
    return out


def progress(method, stage, **details):
    atomic(V2 / 'status' / f'train.{method}.json',
           {'state': stage, 'at': now(), 'method': method, 'pid': os.getpid(), **details})


def train(method):
    rows = records(method, 'expert_train')
    output = V2 / 'adapters/0.6B' / method / 'seed42'
    output.mkdir(parents=True, exist_ok=True)
    rank = 80 if method == 'mix80' else 16
    config = {'method': method, 'scale': '0.6B', 'rank': rank, 'alpha': 2*rank,
              'dropout': .05, 'seed': 42, 'epochs': 2, 'micro_batch': 1,
              'gradient_accumulation': 16, 'lr': 1e-4, 'betas': [.9, .999],
              'eps': 1e-8, 'weight_decay': .01, 'warmup_ratio': .05,
              'max_grad_norm': 1.0, 'max_seq': 2048, 'precision': 'bfloat16',
              'source_ids': [r['id'] for r in rows],
              'teacher_shas': [r['record_sha'] for r in rows],
              'manifest_sha256': digest(PILOT), 'script_sha256': digest(__file__)}
    cfgpath = output / 'run.json'
    if cfgpath.exists():
        assert json.loads(cfgpath.read_text()) == config, 'Run config changed on resume'
    else:
        atomic(cfgpath, config)
    complete = output / 'complete.json'
    if complete.exists():
        return json.loads(complete.read_text())
    seed_all(42)
    model = attach(load_model('0.6B'), rank)
    geo = geometry(model, rank)
    model.train()
    model.config.use_cache = False
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    params = [p for p in model.parameters() if p.requires_grad]
    frozen = {n: p._version for n, p in model.named_parameters() if not p.requires_grad}
    optimizer = torch.optim.AdamW(params, lr=1e-4, betas=(.9, .999), eps=1e-8, weight_decay=.01)
    total_steps = math.ceil(len(rows)/16)*2
    warmup = math.ceil(total_steps*.05)

    def lr_lambda(step):
        if step < warmup:
            return step/max(1, warmup)
        return .5*(1+math.cos(math.pi*(step-warmup)/max(1, total_steps-warmup)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    step = 0
    start_epoch = 0
    start_offset = 0
    tokens = 0
    response_tokens = 0
    exposures = collections.Counter()
    elapsed = 0.0
    nonzero = 0
    resume = output / 'resume.pt'
    if resume.exists():
        checkpoint = torch.load(resume, map_location='cpu', weights_only=False)
        assert checkpoint['config_sha'] == digest(cfgpath)
        set_peft_model_state_dict(model, checkpoint['adapter'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        step = checkpoint['step']
        start_epoch = checkpoint['epoch']
        start_offset = checkpoint['offset']
        tokens = checkpoint['tokens']
        response_tokens = checkpoint['response_tokens']
        exposures.update(checkpoint['exposures'])
        elapsed = checkpoint['elapsed']
        nonzero = checkpoint['nonzero']
        torch.set_rng_state(checkpoint['rng_cpu'])
        torch.cuda.set_rng_state_all(checkpoint['rng_cuda'])
        del checkpoint
    t0 = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    progress(method, 'running', step=step, total_steps=total_steps,
             exposed=len(exposures), requested=len(rows))
    for epoch in range(start_epoch, 2):
        ordered = order(rows, 42, epoch, method.startswith('mix'))
        offset = start_offset if epoch == start_epoch else 0
        for base in range(offset, len(ordered), 16):
            if stop:
                progress(method, 'paused', step=step, total_steps=total_steps)
                return None
            chunk = ordered[base:base+16]
            loss_sum = 0.0
            for row in chunk:
                x, y = batch_record(row, model.device)
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    loss = model(input_ids=x, attention_mask=torch.ones_like(x),
                                 labels=y, use_cache=False).loss
                assert torch.isfinite(loss).item(), f'nonfinite loss {row["id"]}'
                (loss/len(chunk)).backward()
                loss_sum += float(loss.detach())
                tokens += x.numel()
                response_tokens += len(row['response_ids'])
                exposures[row['id']] += 1
            norm = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
            assert torch.isfinite(norm).item()
            if float(norm) > 0:
                nonzero += 1
            assert any(p.grad is not None for p in params)
            used_lr = optimizer.param_groups[0]['lr']
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            assert all(p._version == frozen[n] and p.grad is None
                       for n, p in model.named_parameters() if n in frozen)
            assert all(torch.isfinite(p).all().item() for p in params)
            seconds = elapsed + time.perf_counter()-t0
            entry = {'at': now(), 'method': method, 'step': step, 'epoch': epoch+1,
                     'offset': base+len(chunk), 'loss': loss_sum/len(chunk),
                     'gradient_norm': float(norm), 'lr': scheduler.get_last_lr()[0],
                     'training_tokens': tokens, 'response_tokens': response_tokens,
                     'seconds': seconds, 'peak_allocated': torch.cuda.max_memory_allocated(),
                     'actual_optimizer_update': bool(used_lr > 0 and float(norm) > 0)}
            checkpoint = {'adapter': {k: v.detach().cpu() for k,v in get_peft_model_state_dict(model).items()},
                          'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                          'config_sha': digest(cfgpath), 'epoch': epoch,
                          'offset': base+len(chunk), 'step': step, 'tokens': tokens,
                          'response_tokens': response_tokens, 'exposures': dict(exposures),
                          'elapsed': seconds, 'nonzero': nonzero,
                          'rng_cpu': torch.get_rng_state(), 'rng_cuda': torch.cuda.get_rng_state_all()}
            torch.save(checkpoint, output/'resume.pt.tmp')
            os.replace(output/'resume.pt.tmp', resume)
            append(V2 / 'logs' / f'train.{method}.jsonl', entry)
            progress(method, 'running', step=step, total_steps=total_steps,
                     exposed=len(exposures), requested=len(rows),
                     training_tokens=tokens, seconds=seconds)
            print(json.dumps(entry), flush=True)
        model.save_pretrained(output/f'epoch{epoch+1}', safe_serialization=True)
        atomic(output/f'epoch{epoch+1}'/'metadata.json',
               {'step': step, 'tokens': tokens, 'config_sha': digest(cfgpath)})
    assert step == total_steps and nonzero > 0
    assert len(exposures) == len(rows) and all(n == 2 for n in exposures.values())
    result = {'state': 'complete', 'steps': step, 'training_tokens': tokens,
              'response_tokens': response_tokens, 'nonzero_gradient_updates': nonzero,
              'exposure_hash': objhash(dict(exposures)), 'exposure_counts': dict(exposures),
              'geometry': geo, 'config_sha': digest(cfgpath),
              'seconds': elapsed+time.perf_counter()-t0}
    atomic(complete, result)
    progress(method, 'trained', step=step, total_steps=total_steps)
    del model, optimizer, scheduler
    gc.collect()
    torch.cuda.empty_cache()
    return result


def select(method):
    rr = records(method, 'adapter_val')
    output = V2 / 'adapters/0.6B' / method / 'seed42'
    assert (output/'complete.json').is_file()
    if (output/'selected.json').exists():
        return json.loads((output/'selected.json').read_text())
    draft = load_model('0.6B')
    domains = DOMAINS if method.startswith('mix') else (method,)
    evidence = []
    cache_dir = output / 'selection_cache'
    cache_dir.mkdir(exist_ok=True)
    cache_manifest = {'validation_ids': [r['id'] for r in rr],
                      'teacher_shas': [r['record_sha'] for r in rr],
                      'adapters': {f'epoch{epoch}': digest(output/f'epoch{epoch}'/'adapter_model.safetensors')
                                   for epoch in (1, 2)},
                      'pilot_sha256': digest(PILOT)}
    cache_binding = output / 'selection_cache_manifest.json'
    if cache_binding.exists():
        assert json.loads(cache_binding.read_text()) == cache_manifest
    else:
        atomic(cache_binding, cache_manifest)

    def validate_diag(diag, reference):
        positions = [j for j in (0,32,64,128,192) if len(reference)-j >= 9]
        assert [w['position'] for w in diag['windows']] == positions
        accepted = []
        for window in diag['windows']:
            assert len(window['proposal']) == 8
            a = 0
            for x,y in zip(window['proposal'], reference[window['position']:]):
                if x != y:
                    break
                a += 1
            assert window['a'] == a
            accepted.append(a)
        assert diag['mean_a'] == (statistics.mean(accepted) if accepted else None)
        assert diag['survival'] == ([sum(a >= k for a in accepted)/len(accepted)
                                     for k in range(1,9)] if accepted else None)
        return diag

    completed = 0
    for epoch in (1,2):
        name = f'epoch{epoch}'
        if epoch == 1:
            draft = PeftModel.from_pretrained(draft, str(output/name),
                                               adapter_name=name, is_trainable=False)
        else:
            draft.load_adapter(str(output/name), adapter_name=name, is_trainable=False)
        draft.set_adapter(name)
        draft.eval()
        by_domain = {d: [] for d in domains}
        cached_audits = set()
        for r in rr:
            cp = cache_dir / f'{epoch}.{hashlib.sha256(r["id"].encode()).hexdigest()}.json'
            if cp.exists():
                saved = json.loads(cp.read_text())
                assert saved['id'] == r['id'] and saved['epoch'] == epoch
                assert saved['cache_manifest_sha256'] == digest(cache_binding)
                diag = validate_diag(saved['diagnostic'], r['response_ids'][:256])
                if r['domain'] not in cached_audits:
                    assert public_prefix(draft, r['prompt_ids'], r['response_ids'][:256], 8) == diag
                    cached_audits.add(r['domain'])
            else:
                diag = validate_diag(public_prefix(draft, r['prompt_ids'],
                                                   r['response_ids'][:256], 8), r['response_ids'][:256])
                atomic(cp, {'id': r['id'], 'domain': r['domain'], 'epoch': epoch,
                            'diagnostic': diag, 'cache_manifest_sha256': digest(cache_binding)})
                append(V2/'logs'/f'select.{method}.jsonl',
                       {'id': r['id'], 'domain': r['domain'], 'epoch': epoch, **diag})
            completed += 1
            if diag['mean_a'] is not None:
                by_domain[r['domain']].append(diag['mean_a'])
            progress(method, 'selecting', completed=completed, expected=2*len(rr),
                     selected_epochs=epoch-1, total_epochs=2)
        assert all(by_domain.values()), 'No qualifying validation windows'
        means = {d: statistics.mean(v) for d,v in by_domain.items()}
        evidence.append({'epoch': epoch, 'domain_means': means,
                         'macro_mean_a': statistics.mean(means.values()),
                         'coverage': {d: len(v) for d,v in by_domain.items()}})
        progress(method, 'selecting', selected_epochs=epoch, total_epochs=2)
    best = max(evidence, key=lambda item: (item['macro_mean_a'], -item['epoch']))
    result = {'selected_epoch': best['epoch'], 'selection_rule': 'macro request mean acceptance L8; ties earlier epoch',
              'evidence': evidence, 'adapter_path': str(output/f'epoch{best["epoch"]}'),
              'partition': 'adapter_val', 'pilot_sha256': digest(PILOT)}
    atomic(output/'selected.json', result)
    progress(method, 'complete', selected_epoch=best['epoch'])
    del draft
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=METHODS, required=True)
    args = parser.parse_args()
    assert_spec()
    torch.set_num_threads(4)
    signal.signal(signal.SIGUSR1, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    trained = train(args.method)
    if trained is not None and not stop:
        select(args.method)


if __name__ == '__main__':
    main()

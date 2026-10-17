"""CPU-only guard tests; safe while the two authorized training jobs use the GPU."""
import contextlib
import json
import pathlib
import tempfile
import unittest
from unittest.mock import patch
import sys
from types import SimpleNamespace
sys.path.insert(0, '/root/autodl-tmp/sd-lora-preexp')
from spec_research.common import assert_spec
assert_spec()
from evaluation_stats import journal, unique_index, paired_delta, aggregate_online, paired_latency_ratio
from continuation_worker import validate_diagnostic
import continuation_evaluate as ev


class Classifier:
    def __init__(self, constant=None): self.constant = constant
    def predict(self, texts): return [self.constant or x for x in texts]


class Guards(unittest.TestCase):
    def test_configuration_driver_timing_residency_reuse_and_fail_closed(self):
        # Synthetic CPU fixture validates orchestration only. It is never saved
        # in the research results or interpreted as measured model performance.
        class Model:
            generation_config = SimpleNamespace(eos_token_id=99)
            active_adapter = None
            def set_adapter(self, x): self.active_adapter = x
            def named_parameters(self): return []
        prompts = [dict(id=d+str(i), domain=d, prompt=d, prompt_ids=[100], response_ids=[1,2,3], record_sha='fixture') for d in ev.DOMAINS for i in range(200)]
        count = dict(ar=0, sd=0)
        def ar_fixture(*args):
            count['ar'] += 1
            return dict(tokens=[1,2,3], latency_ms=30., ttft_ms=10., decode_seconds=.020, target_calls=3, margins=[])
        def sd_fixture(*args):
            count['sd'] += 1
            return dict(tokens=[1,2,3], latency_ms=20., ttft_ms=5., decode_seconds=.015,
                        rounds=[dict(ar=2, Lr=2, nr=3)], target_calls=2, target_prefill_calls=1, target_verify_calls=1, draft_calls=2, margins=[])
        with tempfile.TemporaryDirectory() as t, contextlib.ExitStack() as stack:
            base = pathlib.Path(t); out = base/'online'/'0.6B'
            router_fixture = base/'router.fixture'; router_fixture.write_bytes(b'CPU fixture only')
            (base/'quality'/'0.6B').mkdir(parents=True)
            (base/'quality'/'0.6B'/'quality_summary.json').write_text('{"fixedbest":"math"}')
            replacements = dict(BASE=base, RECOVERY=base/'recovery', router_path=lambda scale:router_fixture, records=lambda:prompts, validate_teacher=lambda rr:rr,
                                bind=lambda *a, **kw:'fixture', check_exclusive=lambda:None, hardware=lambda:{'fixture':True},
                                load_model=lambda scale:Model(), pool=lambda scale, methods:Model(), weight_bytes=lambda model:1,
                                partition=lambda d,p:[r for r in prompts if r['domain'] == d], ar=ar_fixture,
                                speculative=sd_fixture, sync=lambda:None, progress=lambda *a, **kw:None, profile_config=lambda *a:None)
            for name, value in replacements.items(): stack.enter_context(patch.object(ev, name, value))
            for name in ('reset_peak_memory_stats',): stack.enter_context(patch.object(ev.torch.cuda, name, return_value=None))
            for name in ('max_memory_allocated','max_memory_reserved','memory_allocated'): stack.enter_context(patch.object(ev.torch.cuda, name, return_value=10))
            ev.ar_config('0.6B', 8, 0)
            self.assertEqual(count['ar'], 610)
            ar_runtime = json.loads((out/'baselines'/'L8.repeat0'/'runtime.json').read_text())
            self.assertEqual(ar_runtime['resident_drafts'], [])
            ev.online_config('0.6B', 'DOMAIN', 8, 0, 1)
            self.assertEqual(count, dict(ar=610, sd=610))
            rows = list(journal(out/'requests.jsonl', base/'recovery'))
            self.assertEqual(len(rows), 600)
            self.assertTrue(all(x['sd']['ttft_ms'] == 5+x['routing_ms'] for x in rows))
            self.assertTrue(all(x['ar_measurement_shared_across_methods'] for x in rows))
            self.assertTrue(all(x['selected'] == x['domain'] for x in rows))
            # The preliminary path is isolated and cannot silently run600.
            before = dict(count)
            ev.ar_config('0.6B',8,0,preliminary=True)
            ev.online_config('0.6B','DOMAIN',8,0,1,preliminary=True)
            self.assertEqual(count['ar']-before['ar'],34)
            self.assertEqual(count['sd']-before['sd'],34)
            preliminary = list(journal(base/'preliminary_online/0.6B/requests.jsonl',base/'recovery'))
            self.assertEqual(len(preliminary),24)
            self.assertTrue(all(sum(r['domain']==d for r in preliminary)==8 for d in ev.DOMAINS))
            self.assertEqual(len(list(journal(out/'requests.jsonl',base/'recovery'))),600)
            original_sd = sd_fixture
            bad_calls = [0]
            def bad(*args):
                value = original_sd(*args); bad_calls[0] += 1
                if bad_calls[0] > 10: value['tokens'] = [1,9,3]
                return value
            with patch.object(ev, 'speculative', bad):
                with self.assertRaisesRegex(RuntimeError, 'Full token mismatch'):
                    ev.online_config('0.6B', 'mix48', 8, 0, 1)
            cfg = out/'configurations'/'L8.repeat0.mix48'
            self.assertTrue((cfg/'failure.json').exists()); self.assertFalse((cfg/'complete.json').exists())
            self.assertEqual(len(list(journal(out/'requests.jsonl', base/'recovery'))), 600)

    def test_legacy_flat_diagnostic_checks_real_first_mismatch(self):
        ref = list(range(60)); proposal = [0, 1, 99, 3, 4, 5, 6, 7]
        d = dict(windows=[dict(position=0, a=2, proposal=proposal)], mean_a=2, survival=[1., 1., 0., 0., 0., 0., 0., 0.])
        # There is also a qualifying window32, so silently omitting it must fail.
        with self.assertRaises(AssertionError): validate_diagnostic(d, ref)
        self.assertEqual(validate_diagnostic(d, ref[:20])['mean_a'], 2)
        bad = dict(d, mean_a=3)
        with self.assertRaises(AssertionError): validate_diagnostic(bad, ref[:20])

    def test_preliminary_sampling_is_frozen_and_independent_of_responses(self):
        prompts=[dict(id=d+str(i),domain=d,prompt='input'+str(i),response_ids=[i]) for d in ev.DOMAINS for i in range(200)]
        with patch.object(ev,'records',return_value=prompts): a=ev.preliminary_records()
        changed=[dict(r,response_ids=[999999]) for r in prompts]
        with patch.object(ev,'records',return_value=changed): b=ev.preliminary_records()
        self.assertEqual([r['id'] for r in a],[r['id'] for r in b])
        self.assertEqual(len({r['id'] for r in a}),24)
        self.assertTrue(all(sum(r['domain']==d for r in a)==8 for d in ev.DOMAINS))

    def test_journal_preserves_interrupted_tail_and_duplicate_conflict(self):
        with tempfile.TemporaryDirectory() as t:
            p = pathlib.Path(t)/'records.jsonl'; rec = pathlib.Path(t)/'recovery'
            raw = b'{"id":"a","value":1}\n{"id":"b"'
            p.write_bytes(raw)
            self.assertEqual(list(journal(p, rec)), [dict(id='a', value=1)])
            self.assertEqual(next(rec.iterdir()).read_bytes(), raw)
            self.assertEqual(p.read_bytes(), raw.split(b'\n')[0]+b'\n')
            p.write_bytes(b'{"id":"a"}')
            self.assertEqual(list(journal(p, rec)), [dict(id='a')]); self.assertTrue(p.read_bytes().endswith(b'\n'))
            p.write_bytes(b'{broken}\n{"id":"a"}\n')
            with self.assertRaises(AssertionError): list(journal(p, rec))
        with self.assertRaises(AssertionError): unique_index([dict(id='x', a=1), dict(id='x', a=2)], lambda x:x['id'])

    def test_paired_bootstrap_never_turns_missing_into_zero(self):
        rr = [dict(domain=d, values=dict(a=5., b=3.)) for d in ev.DOMAINS]
        rr.append(dict(domain='math', values=dict(a=None, b=100.)))
        result = paired_delta(rr, 'a', 'b')
        self.assertEqual(result['macro_delta'], 2); self.assertEqual(result['ci95'], [2., 2.])
        self.assertEqual(result['covered_requests']['math'], 1)

    def test_direct_latency_pairing_is_by_request_id(self):
        rr = [dict(id=d+str(i),domain=d,repeat=0,L=8,method=m,sd_ms=ms) for d in ev.DOMAINS for i in range(200)
              for m,ms in [('mix16',20.+i),('ROUTE-ACC',(20.+i)/2)]]
        rr.reverse()
        result = paired_latency_ratio(rr,'ROUTE-ACC','mix16',0,8)
        self.assertAlmostEqual(result['macro_speedup_geo'], 2.)
        self.assertAlmostEqual(result['ci95'][0], 2.)
        with self.assertRaises(AssertionError): paired_latency_ratio(rr[1:],'ROUTE-ACC','mix16',0,8)

    def test_ratio_of_sums_and_committed_bonus(self):
        rr = []
        for d in ev.DOMAINS:
            for i, (ar_ms, sd_ms) in enumerate([(100., 50.), (10., 20.)]):
                tokens = [1, 2, 3]
                sd = dict(tokens=tokens, latency_ms=sd_ms, ttft_ms=2., decode_seconds=.01,
                          rounds=[dict(ar=1, Lr=2, nr=2), dict(ar=0, Lr=1, nr=1)],
                          target_calls=3, target_prefill_calls=1, target_verify_calls=2, draft_calls=3)
                rr.append(dict(id=d+str(i), domain=d, L=8, ar=dict(tokens=tokens, latency_ms=ar_ms), sd=sd,
                               routing_ms=1., token_match=True, peak_allocated=10, peak_reserved=20))
        result = aggregate_online(rr)
        self.assertAlmostEqual(result['macro_speedup_geo'], 110/70)
        self.assertAlmostEqual(result['draft_acceptance_rate'], 1/3)
        self.assertEqual(result['mean_committed_per_round'], 1.5)
        self.assertAlmostEqual(result['confirmed_tokens_per_s'], 18/.210)
        self.assertEqual(result['output_mismatch_rate'], 0.)
        rr[0]['sd']['tokens'] = [1, 9, 3]
        with self.assertRaises(AssertionError): aggregate_online(rr)

    def test_residency_and_prompt_only_choice(self):
        self.assertEqual(ev.resident_methods('BASE', 'math'), [])
        self.assertEqual(ev.resident_methods('mix48', 'math'), ['mix48'])
        self.assertEqual(ev.resident_methods('FIXEDBEST', 'code'), ['code'])
        self.assertEqual(ev.resident_methods('ROUTE-ACC', 'math'), list(ev.DOMAINS))
        r = dict(prompt='code', domain='math', response_ids=[100, 200])
        self.assertEqual(ev.choose('ROUTE-ACC', r, 'general', Classifier()), 'code')
        self.assertEqual(ev.label(dict(math=3., code=3., general=1.), 'code'), 'code')
        self.assertEqual(ev.label(dict(math=3., code=3., general=1.), 'general'), 'math')

    def test_matrix_oracle_fixed_router_and_short_answer_coverage(self):
        with tempfile.TemporaryDirectory() as t:
            base = pathlib.Path(t); out = base/'quality'/'0.6B'; out.mkdir(parents=True)
            prompts = []; entries = []
            score = {'math': [6.,1.,2.], 'code': [5.,4.,1.], 'general': [2.,3.,1.]}
            for d in ev.DOMAINS:
                for i in range(200):
                    r = dict(id=d+str(i), domain=d, prompt=d); prompts.append(r)
                    for m in ['BASE', *ev.METHODS]:
                        value = 2. if m == 'BASE' else (4. if m.startswith('mix') else score[d][ev.DOMAINS.index(m)])
                        diag = dict(mean_a=value, survival=[float(k<=value) for k in range(1,9)], windows=[dict(position=0,a=int(value))])
                        if i == 0: diag = dict(mean_a=None, survival=None, windows=[])
                        entries.append(dict(id=r['id'], domain=d, method=m, L=8, diagnostic=diag))
            (out/'matrix.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in entries))
            acc_path = base/'acc.pkl'; acc_path.touch()
            with patch.object(ev, 'BASE', base), patch.object(ev, 'records', return_value=prompts), patch.object(ev, 'router_path', return_value=acc_path), patch.object(ev, 'classifier', side_effect=lambda p: Classifier('math') if p == acc_path else Classifier()):
                ev.quality_report('0.6B')
            result = json.loads((out/'quality_summary.json').read_text())
            self.assertEqual(result['state'], 'complete'); self.assertEqual(result['qualified_requests'], 597)
            self.assertEqual(result['fixedbest'], 'math')
            self.assertAlmostEqual(result['methods']['ORACLE']['macro_mean_a'], 14/3)
            self.assertAlmostEqual(result['methods']['DOMAIN']['macro_mean_a'], 11/3)
            self.assertAlmostEqual(result['methods']['mix48']['macro_mean_a'], 4.)
            self.assertAlmostEqual(result['routing']['ROUTE-ACC']['oracle_regret'], 1/3)
            self.assertEqual(result['coverage']['math']['no_public_window'], 1)


if __name__ == '__main__': unittest.main(verbosity=2)

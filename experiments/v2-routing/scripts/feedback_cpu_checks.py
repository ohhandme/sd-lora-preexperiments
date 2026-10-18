#!/root/miniconda3/envs/spec/bin/python
import pathlib
import sys

ROOT = pathlib.Path('/root/autodl-tmp/sd-lora-preexp')
sys.path[:0] = [str(ROOT), str(ROOT / 'experiments/v2-routing/scripts')]
from feedback_online import Policy
from train import DOMAINS

cost = {'offline_mix80_mean_a': 3.8,
        'target_verify_8_ms_median': 150.0,
        'adapter_switch_ms_median': 6.0}
prior = {d: (5.0 if d == 'code' else 3.0) for d in DOMAINS}
meta = Policy('META-UCB', 'code', prior, cost)
assert meta.choose(0, 100, 100) == 'code'
meta.observe('code', 4)
assert meta.choose(1, 104, 100) == 'translation'
assert meta.switches == 1
for d in DOMAINS[1:]:
    meta.observe(d, 3)
assert meta.choose(5, 120, 100) in DOMAINS
meta.switches = 8
current = meta.current
assert meta.choose(6, 120, 100) == current
safe = Policy('COST-UCB', 'code', prior, cost)
assert safe.choose(0, 100, 100) == 'code'
safe.first_draft_prefill_ms = 1000.0
safe.observe('code', 4)
assert safe.choose(1, 110, 100) == 'code'
print('feedback policy CPU checks passed')

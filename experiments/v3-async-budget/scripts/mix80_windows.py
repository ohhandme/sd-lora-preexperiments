#!/root/miniconda3/envs/spec/bin/python
"""Generate missing MIX80 public-prefix diagnostics on frozen router_train rows.

Only calibration/holdout request IDs from V3 are touched. No teacher outputs
or V2 results are modified; all records are checked against the saved teacher.
"""

import hashlib
import json
import os
import pathlib
import sys

ROOT = pathlib.Path('/root/autodl-tmp/sd-lora-preexp')
V2 = ROOT/'experiments/v2-routing'
V3 = ROOT/'experiments/v3-async-budget'
sys.path[:0] = [str(ROOT),str(V2/'scripts'),str(V3/'scripts')]

import torch
from peft import PeftModel
from spec_research.common import assert_spec,atomic,digest,load_model,now
from spec_research.decoding import public_prefix
from train import records
from offline_probe import split_ids

OUT = V3/'results/mix80_router_train'


def main():
    assert_spec();torch.set_num_threads(4)
    rows=records('mix16','router_train');splits=split_ids(rows)
    chosen=sorted((r for r in rows if r['id'] in splits['calibration']|splits['holdout']),
                  key=lambda r:r['id'])
    assert len(chosen)==190
    selected=json.loads((V2/'adapters/0.6B/mix80/seed42/selected.json').read_text())
    adapter=pathlib.Path(selected['adapter_path'])
    binding={'ids':[r['id'] for r in chosen],
             'pilot_sha256':digest(V2/'data/processed/pilot_manifest.jsonl'),
             'adapter_sha256':digest(adapter/'adapter_model.safetensors'),
             'script_sha256':digest(__file__),
             'policy':'same BF16 L8 positions 0,32,64,128,192 as v2 public_prefix'}
    manifest=V3/'status/mix80_input.json'
    if manifest.exists():assert json.loads(manifest.read_text())==binding
    else:atomic(manifest,binding)
    model=PeftModel.from_pretrained(load_model('0.6B'),adapter,adapter_name='mix80',is_trainable=False).eval()
    for index,row in enumerate(chosen,1):
        path=OUT/(hashlib.sha256(row['id'].encode()).hexdigest()+'.json')
        if path.exists():
            item=json.loads(path.read_text())
            assert item['id']==row['id'] and item['binding']==digest(manifest)
            continue
        diag=public_prefix(model,row['prompt_ids'],row['response_ids'][:256],8)
        for w in diag['windows']:
            ref=row['response_ids'][w['position']:w['position']+8]
            a=next((j for j,(x,y) in enumerate(zip(w['proposal'],ref)) if x!=y),8)
            assert a==w['a'] and len(ref)==8
        atomic(path,{'id':row['id'],'domain':row['domain'],'binding':digest(manifest),
                     'teacher_sha256':row['record_sha'],'diagnostic':diag})
        atomic(V3/'status/mix80_windows.json',{'state':'running','at':now(),
               'pid':os.getpid(),'completed':index,'total':190})
    atomic(V3/'status/mix80_windows.json',{'state':'complete','at':now(),
           'completed':190,'total':190,'binding_sha256':digest(manifest)})
    print(json.dumps({'state':'complete','total':190,'binding_sha256':digest(manifest)}))


if __name__=='__main__':main()

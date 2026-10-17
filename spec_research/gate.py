"""Independent fail-closed guard over immutable P0 evidence."""
import math
from .common import *
from .data import smoke_records,validate_data
from .teacher import validate_teacher

def validate_gate(path=None):
    assert_spec();path=path or ROOT/'manifests/p0_gate.json';g=read(path)
    assert g['state']=='passed' and g['prefix']==SPEC and g['mismatches']==0
    assert g['source_hashes']==sources(),'Code/config changed after P0'
    for p,sha in g['evidence_hashes'].items():assert digest(ROOT/p)==sha,p
    validate_data();rr=smoke_records();ids=[r['id'] for r in rr]
    assert len(ids)==192 and len(set(ids))==192
    validate_teacher(rr)
    report=read(ROOT/g['report']);assert report['state']=='passed' and report['ids']==ids
    assert report['provenance']['source_hashes']==sources()
    env=read(ROOT/'results/smoke/environment.json');assert env['state']=='passed'
    assert set(env['stages']['models'])=={'8B','0.6B','4B'}
    assert set(env['stages']['tokenizers'])=={'8B','0.6B','4B'}
    assert env['stages']['gpu_kernels']['observed_ops']
    for scale in ('0.6B','4B'):
        checks=env['stages']['models'][scale]
        assert checks['adapter_updated'] and checks['base_unchanged'] and checks['assistant_shifted_ce_checked'] and checks['gradient_norm']>0
        assert set(checks['resident_activations'])=={'default','code','general','mix48'}
        assert checks['r48']['adapter_params']==3*checks['r16']['adapter_params']
        assert checks['cached_ar_vs_hf']['token_match']

    tests=read(ROOT/'results/smoke/scripted.json');assert tests['exit_code']==0 and tests['tests']>=54
    for scale in ('0.6B','4B'):
        sm=report['scales'][scale];train=read(sm['training_complete'])
        assert train['state']=='complete' and train['nonzero_gradient_updates']>0 and train['steps']==12
        log=ROOT/'logs/smoke'/f"{report['run_id']}.{scale}.train.jsonl"
        entries=list(rows(log));assert len(entries)==12
        assert all(math.isfinite(e['loss']) and math.isfinite(e['gradient_norm']) and e['gradient_norm']>0 for e in entries)
        assert any(e['actual_optimizer_update'] for e in entries)

        assert train['exposure_counts']=={i:1 for i in ids}
        primary=list(rows(pathlib.Path(sm['primary'])/'requests.jsonl'))
        assert len(primary)==192 and [r['id'] for r in primary]==ids
        assert all(r['token_match'] and r['L']==8 and r['max_new_tokens']==64 for r in primary)
        extra=list(rows(pathlib.Path(sm['extra'])/'requests.jsonl'))
        assert len(extra)>=6 and {r['L'] for r in extra}=={4,12} and all(r['token_match'] for r in extra)
    return g

def create_gate(report_path):
    assert not (ROOT/'manifests/p0_gate.json').exists(),'Gate is immutable; archive prior gate explicitly before a new verification'
    report=read(report_path);assert report['state']=='passed'
    files=[report_path,ROOT/'results/smoke/environment.json',ROOT/'results/smoke/scripted.json',ROOT/'manifests/data.json',ROOT/'manifests/environment/spec-pip-freeze.txt']
    files.extend((ROOT/'results/smoke').rglob('*.jsonl'))
    files.extend(ROOT/'logs/smoke'/f"{report['run_id']}.{scale}.train.jsonl" for scale in ('0.6B','4B'))
    for sm in report['scales'].values():
        files.append(pathlib.Path(sm['training_complete']))
        files.extend(pathlib.Path(sm['adapter']).rglob('*.safetensors'))
    files.extend(ROOT/'data/teacher'/r['domain']/(objhash(r['id'])+'.json') for r in smoke_records())
    evidence={str(p.relative_to(ROOT)):digest(p) for p in set(files)}
    gate={'state':'passed','at':now(),'prefix':sys.prefix,'source_hashes':sources(),'evidence_hashes':evidence,'report':str(pathlib.Path(report_path).relative_to(ROOT)),'mismatches':0,'commands':report['commands'],'stages':report['stages'],'provenance':provenance()}
    path=ROOT/'manifests/p0_gate.json';atomic(path,gate)
    try:validate_gate(path)
    except Exception:
        # Preserve failed evidence. Never leave a failed guard claiming passed.
        gate['state']='guard_failed';atomic(path,gate);raise
    path.chmod(0o444);return gate
